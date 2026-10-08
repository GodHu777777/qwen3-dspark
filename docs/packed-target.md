# 可变 query 长度的单次 Qwen 前向参考

`packed_target.py` 把多个请求的新片段拼成 `[1, sum(query_lengths)]`，一次真实
Qwen3 backbone forward 处理所有片段。它没有按最长请求补 query padding，也没有
逐请求调用模型。词表投影同样一次处理所有新行，再按 spans 返回各请求结果。
这是请求状态与 attention 隔离的 **dense SDPA 参考实现**，不是完整 serving
engine、高效 varlen kernel、scheduler、CUDA/ROCm graph 或性能验证。

## HF 5.17 接口与隔离

读取当前固定的 Transformers 5.17 Qwen3 源码后，确认无需定制 decoder layer：

1. `position_ids` 显式使用各请求自己的连续位置，而不是物理拼接后的序号。
2. `attention_mask={'full_attention': mask}` 绕过 Qwen3 默认 mask 构造，
   避免把 DynamicCache 的物理长度当成所有请求的语义 prefix 长度。
3. 每个 query/key 保存 request marker 和 request-local position。
   可见条件为 `q.request == k.request and k.position <= q.position`。
4. HF DynamicCache 每层沿物理 token 轴追加拼接 KV。各请求的 KV 可以交错存放，
   不要求某请求占据一段连续物理区间。

仅支持 dense Qwen3、full-attention layers、SDPA、default RoPE。
拒绝 sliding-window 和动态 RoPE；后者可能根据 batch 最大 position 改变频率，
从而让较长请求影响较短请求，不能未经验证继承隔离结论。该适配依赖 HF 5.17 的
字典 mask 和 DynamicLayer `.keys/.values` 接口，依赖升级需要重测。

## 状态与调用

```python
target = PackedTarget(model, layer_ids=(1, 7, 14, 21, 26))
target.add_request('a')
target.add_request('b')
features = target.append({'a': a_prompt_ids, 'b': b_prompt_ids})
logits_by_request = target.predict(features)

# 下一轮可以只包含部分活跃请求，也可以包含不同长度的验证片段。
features = target.append({'b': b_anchor_and_draft, 'a': a_anchor_and_draft})
target.crop('a', committed_a_prefix_length)
target.crop('b', committed_b_prefix_length)
target.remove_request('b')  # 调用方确认 EOS/退出后释放
target.add_request('c')
```

所有 token tensor 必须为模型所在设备上的非空 `[1,q]` long。
`append` 不自动新建请求；`add_request` 允许空前缀，`crop` 可保留到 0 后继续追加。
`remove_request` 清除该请求所有层的 KV，外部 ID 再次加入时分配新 marker，
从本地 position 0 开始。inactive resident 请求本轮没有 query，仍保留 KV。

`crop(request_id, length)` 是请求内部绝对前缀长度，不是全局物理长度。
它按 request/position mask 对每层 K/V 作 index_select，先构造所有层的新 tensor，
再统一替换，保留其它请求的内容与物理顺序。这里是分配/拷贝式 gather，
不是 paged-KV allocator 或 graph-stable 内存管理。

EOS、随机接受/残差采样、输出预算与 draft KV 均由调用方负责；本模块只提供目标
状态操作，不能靠目标 crop 自动修复 draft cache。输入校验失败不改变缓存；
模型在前向中途失败可能仅扩展部分层，此时全部请求状态清空，禁止复用半成品。
`request_kv` 返回各请求本地顺序的审计副本，适合内容比对，不适合性能路径。

## Query 行数与 attention 工作域

`features.work` 保存 query 长度、query 前各请求 context 长度、全部 resident
context 长度，以及 queried/resident 请求数，供以后设计 `SPS(B, context/R)`
profile 使用。此模块不会自动给每请求添加 anchor；调用方传入的 query 若包含
anchor，anchor 已计入 Q。与 scheduler 的 `B = sum(1 + ell_r)` 对接时必须显式
包含各请求 baseline 行。

| 字段 | 含义 |
| --- | --- |
| `logical_query_tokens`、`physical_query_tokens` | 二者均为 Q = 所有新片段长度之和，无 query padding |
| `physical_kv_tokens_before`、`physical_kv_tokens` | 调用前/后共享缓存的总行数；后者 K 包含 inactive resident keys |
| `dense_attention_pairs_per_head_layer` | dense score 域 Q×K；不是硬件逐元素运算计数 |
| `allowed_causal_pairs` | 同请求且不晚于 query position 的可见元素数 |
| `masked_cross_request_pairs` | 跨请求（包括 inactive resident）被 mask 的 score 域元素数 |
| `masked_future_pairs` | 同请求但未来位置被 mask 的元素数 |
| `mask_shape`、`mask_bytes` | 实际显式 boolean mask `[1,1,Q,K]` 的形状与存储大小 |

三个 pair 字段之和等于 Q×K。还保存 head/layer 数与全 head/layer score 域大小。
SDPA 后端可能融合或以不同 tile 执行，不能把这些数字说成实测 FLOPs，也不声称
底层一定逐个计算每个 mask 元素。当前没有提供 block-sparse/FlexAttention/varlen
kernel 的跳过证据，因此不能把物理 query 无 padding 描述为 attention 已经高效。

例如 3 个新请求 query 长度 `[3,1,4]`，Q=8（不是 3×4=12），K=8，dense score
域为 64，因果可见数为 6+1+10=17。随后只给 b 追加 3 行、a 追加 1 行，c 的
4 行 KV 本轮 inactive，Q=4、K=12，score 域仍为 48；cross-request 计数必须
包括这些 idle keys，而不能只看当前 query 请求。

## CPU 验证与边界

运行 `python -m unittest discover -s tests -p test_packed_target.py -v`。
实际执行环境 Torch 2.12.0+rocm7.2 / Transformers 5.17.0，显式 CPU、FP32、
随机小 Qwen3（vocab64、hidden32、3 layers、4 query heads/2 KV heads）。
本地原始证据位于 ignored `output/packed-target-cpu-20261009/`。

五项检查覆盖：

- 混合长度与 query 顺序变化、inactive resident keys；forward hook 证明每轮
  仅一次真实模型调用，输入 physical Q 与每请求重置 position_ids 均正确。
- 与独立每请求 CachedTarget 的 chunk hidden/context/logits 逐项比较，并与
  fresh full-prefix forward 比较 hidden；每一层 K/V 内容同时对照 cached/fresh
  两个 oracle，不能仅检查 cache length。
- 全拒绝、部分保留、每个 crop 边界、所有请求 crop 到 0 后继续追加。
- EOS/退出由调用方释放状态、移除后相同外部 ID 重新加入、新旧请求交错。
- 把另一请求的 K/V 改为大幅有限 poison，并改变其新 token 和同请求未来 token，
  当前请求的更早 query hidden/context/logits 仍隔离；输入异常保留状态，前向中途
  异常清空所有状态。

比较容差为 CPU FP32 `atol=2e-6, rtol=1e-5`，poison 检查 atol=1e-6。
这不证明 AMD BF16 或不同 kernel/batch shape 的 token 数值等价；原先动态 BF16
失败仍未被覆盖。没有读取真实数据/最终 test，没有测接受率、吞吐、速度或 SLA。

后续高效执行仍需选择支持目标 ROCm/hardware 的 varlen 或 block-sparse attention，
以相同请求状态 oracle 检查隔离，并实测物理 mask/tile 工作、KV gather 代价、
调度和执行重叠。FlexAttention 或其它 kernel API 的存在本身不等于 gfx1201 支持
及速度收益；该核验不在当前 CPU 里程碑内。
