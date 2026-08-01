# 优化记录:groupby 直发(direct-send)+ 窄接收表(narrow receiver)

日期:2026-07-31。提交:Magi-Dev `821504b0` / sirius `a82df116`(依赖前置的 `SendDirectStream` API,Magi-Dev `9fcde08c`)。

## 动机

大档位(≥XLARGE)grouped aggregate 的管线是:

```
cudf 本地预聚合(每卡去重到每 key 一行)
→ kernel A: global_preagg 把这些行插进 producer 哈希 H   ← 冗余
→ kernel B: 扫 H 的活槽 → shuffle 到 owner GPU → 接收表合并
→ compact(含 HAVING 下推)→ 输出
```

输入既然已被 cudf 去过重,H 的每个 key 只插一次、什么都不合并 —— 它退化成纯"发送前暂存区",却要付:每查询 16GB 表初始化(XXXLARGE 两张 8GB arena)、7500 万次随机原子插入(Q18@SF100)、1.28 亿槽稀疏扫描。此外这 16GB 是**常驻** arena,把 cache 地板抬到 20.9GB,SF100 下挤掉 Q1 的 8.9GB 列(溢到 host cache,Q1 慢 2×),逼出过 `MAGI_GROUPBY_XXXL` 手动开关。

## 改动

### ① 直发(shuffle_direct)

预聚合档位跳过 H:`pack_preagg_row_tuple` 把一行预聚合输入直接打包成 wire tuple(逐 AggKind 与 producer 写 H 的字节**完全一致**:SUM_INT64 原始位、MIN/MAX 保序编码 e/~e、COUNT 为 u64),`shuffle_direct_partials_ap` 经 `SendDirectStream` 流式上线;接收端不变。接收/合并环重构成 Applier 回调(`receiver_merge_until_eof_ap` + 胖表包装),两种接收表共享同一套 EOF/flush 协议。

- 非预聚合路径(SMALL 的 shmem 单 kernel、MEDIUM/LARGE 的原始行 global-H)**不动** —— 那里 producer 聚合就是压缩上线量的价值本体(Q1:1.48 亿行 → 4 组)。
- 开关:`MAGI_NO_DIRECT_SEND=1` 强制旧路径(A/B 与逃生)。

### ② 窄接收表(XXXLARGE,≤8B key)

胖 64B 槽表(key+6 值同槽,128M×64B=8GB 常驻)拆成:

- `keys[128M]`:8B/槽,CAS 目标,全 1 哨兵 —— 1GB
- `values[128M × n_vals]`:**按查询实际值槽数**(ops 表 max dst+1)步长,Q18 只有 2 槽 → 2GB

两者都是**每查询**从 processing 池分配,用完即还,常驻为零。无 payload 下标、无 bump 计数器、无发布 fence:值就在 key 的槽下标处,逐 kind 原子更新沿用零初始化兼容的编码。`narrow_compact_to_slots` 把活槽重建成密集 AggSlot64 前缀(**HAVING 下推内联**;幸存者超过 scratch 容量时置位回落而非截断),下游 flush/emit 全链路零改动。

配套:`PickTableSize(allow_xxxl)` 只对"预聚合 + 非 u128 key"放行 XXXLARGE;`MAGI_GROUPBY_XXXL` **默认关闭**,arena 回 4+4GB(floor 20.9→12.9GB),该 env 仅为残余胖 XXXLARGE 形态(u128 key / 强制旧路径)重新扩容。

## 实测(2×H100,协议:fresh session/query,warm=min(run2,run3))

| 项 | 旧 | 新 | 备注 |
|---|---|---|---|
| Q18@SF100 qt(52/16/90)| 152.9ms | **120.6ms(−21%)** | 行集与旧路径/CPU digest 一致 |
| └ phase:init | 16.2ms | ~1.5ms | 16GB 表初始化 → keys kernel + values memset |
| └ phase:kernel A | 13.0ms | 0 | 整个跳过 |
| └ phase:kernel B | 36.3ms | ~53ms(直发)| 列读取从 A 挪入 B(pack gather),非新增成本 |
| Q1@SF100 qt(46/30/90)| 651ms(需 `XXXL=0` 才 313)| **314ms,无任何开关** | floor 回 12.9GB,列不再溢 host |
| SF100 全量(fuse 开)| — | 零真实退化 | Q9 156(节点带内)、其余 ±5ms |
| SF50 全量(fuse 开)| — | 零真实退化 | 22 条全部 ±1.5ms |

## 实现要点 / 教训

1. **打包器必须逐字节复刻 producer 语义**(保序 MIN/MAX 编码、SUM_INT64 走原始位而非 double),接收端才能对两种来源无感。
2. 值数组最初按最大 6 槽分配(6GB)触发 proc 池 OOM → 改按查询实际 `n_vals`,Q18 降到 2GB。
3. 微基准里孤立测得的窄表插入税(约 +65%/插入 kernel)在真实流水线中被 channel 等待气泡吸收,端到端反而 −21% —— 孤立微基准适合定机制上限,端到端结论必须实测。
4. compact 的 HAVING 必须内联进窄表 compact(先密集再过滤会让无 HAVING 的 7500 万幸存者撑爆 4GB scratch);容量守卫保证超限走回落而非静默截断。
5. 窄分支用 `if constexpr` 门控(`sizeof(KeyT)<=8 && N_SLOTS==XXXLARGE && SB==64`),避免无关实例化引用窄 kernel。

## 后续(backlog)

- **result conversion**:`.timer` 与 `[query-time]` 的差主要是 GPU 结果 → duckdb DataChunk 的逐行转换,输出大时占绝对主导(Q14@SF50:qt 155ms vs .timer 1.46s,310 万行);批量列式转换是下一块肉。
- int128 SUM(Q1 sum_charge 回绕、Q11 逐字版)—— 做完 SF100 才是干净的 22/22。
- join 发送路径迁 `SendDirectStream`(顺带审计同款无界写隐患,shuffle v1 75M 停摆嫌疑)。
- MEDIUM/LARGE 是否也值得 cudf 预聚合 + 直发:表小、init 便宜,预计收益零附近,要做先拿数据。
