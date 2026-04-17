可以，但你要先决定你说的“换成 Qwen3.5 架构”到底是：

1. **做一个 MiniMind 风格的“Qwen3.5-lite 文本版”**
   也就是保留现在项目的训练 / 推理框架，只把核心 block 改成更像 Qwen3.5 的结构。这个最现实。

2. **完整复刻 Qwen3.5**
   包括混合注意力、Gated DeltaNet、Gated Attention、部分维度 RoPE / MRoPE、可能还有多模态 ViT + PatchMerger。这个工作量会大很多。

按你现在这份 `model_minimind.py` 看，MiniMind 目前还是一个比较标准的 decoder-only 架构：

* 统一的 `Attention` + `FeedForward/MOEFeedForward` block 
* RoPE 是标准整头旋转，不是 Qwen3.5 那种“只旋转前 25% 维度”
* 每层都是同一种 attention，没有 Qwen3.5 的 3:1 混合块

而你给的说明里，Qwen3.5 的关键差异是：

* **3 层线性注意力 + 1 层全注意力** 的混合堆叠
* 全注意力里 **q_proj 要拆成 query + gate**，并对 Q/K 做 **Zero-Centered RMSNorm** 
* 只对前 **25% 维度** 做旋转位置编码，且 Qwen3.5 文档里还提到了 MRoPE 思路 
* MoE 版本不是你现在这种纯 router+experts，而是 **router experts + shared expert** 

所以，**最合理的路线不是“直接替换 Attention 类”，而是分 4 步改**。

---

## 一、先做一个可跑的 Qwen3.5-lite 文本版

先别上多模态，先把文本 backbone 改像 Qwen3.5。

### 第一步：把 block 改成“可切换 attention 类型”

你现在 `MiniMindBlock` 写死了：

```python
self.self_attn = Attention(config)
```

应该改成：

```python
if config.attn_type == "full":
    self.self_attn = Qwen35GatedAttention(config)
elif config.attn_type == "linear":
    self.self_attn = Qwen35GatedDeltaNet(config)
else:
    raise ValueError(...)
```

然后在 `MiniMindModel` 里不要再直接：

```python
self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
```

而是按层号生成模式，比如 3:1：

```python
def get_layer_attn_type(layer_id):
    return "full" if (layer_id + 1) % 4 == 0 else "linear"
```

这一步是最核心的，因为 Qwen3.5 的 backbone 不是“统一 attention block”，而是**交错混合 block**。你的 `index.md` 里也明确写了 3:1 混合结构 。

---

## 二、先实现 Gated Attention，再考虑 Gated DeltaNet

### 先改全注意力最划算

你现在的 `Attention` 是普通 GQA/MQA 风格：

* `q_proj / k_proj / v_proj / o_proj`
* `q_norm / k_norm`
* RoPE 后算 attention 

但 Qwen3.5 的全注意力至少要补两个东西：

### 1）`q_proj` 输出要拆成 `q + gate`

你现在：

```python
self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
```

Qwen3.5 风格应该更像：

```python
self.q_proj = nn.Linear(
    config.hidden_size,
    2 * config.num_attention_heads * self.head_dim,
    bias=False
)
```

前向里：

```python
q_and_gate = self.q_proj(x)
q, gate = q_and_gate.chunk(2, dim=-1)
```

attention 输出后再门控：

```python
attn_out = attn_out * torch.sigmoid(gate)
```

你的说明里写得很明确：**Gate 用于在注意力输出后应用查询依赖门控** 。

### 2）把 `RMSNorm` 改成 `ZeroCenteredRMSNorm`

你现在的 `RMSNorm` 只是：

```python
return (self.weight * self.norm(x.float())).type_as(x)
```

Qwen3.5 里更接近：

```python
class ZeroCenteredRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        x = x - x.mean(dim=-1, keepdim=True)
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * rms * self.weight).type_as(x)
```

然后替换 attention 里的 `q_norm`、`k_norm`。
你给的笔记明确说 Qwen3.5 对 Q/K 使用 **Zero-Centered RMSNorm** 来提升稳定性 。

---

## 三、把 RoPE 改成“部分旋转”，不要再整头旋转

这是你当前实现和 Qwen3.5 差异非常大的地方。

你当前 `apply_rotary_pos_emb` 是对整个 head_dim 全旋转的 。
而 Qwen3.5 说明里写得很清楚：**只对前 25% 的维度旋转**，例如 head_dim=256 时，只旋转 64 维 。

你可以这样改：

### config 里加参数

```python
self.partial_rotary_factor = kwargs.get("partial_rotary_factor", 0.25)
self.rotary_dim = int(self.head_dim * self.partial_rotary_factor)
```

### 预计算频率时只按 rotary_dim 算

把：

```python
freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.head_dim, ...)
```

改成：

```python
freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.rotary_dim, ...)
```

### rotary 应用时只旋转前半段

```python
def apply_partial_rotary_pos_emb(q, k, cos, sin, rotary_dim, unsqueeze_dim=1):
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]), dim=-1)

    q_rot = (q_rot * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q_rot) * sin.unsqueeze(unsqueeze_dim))
    k_rot = (k_rot * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k_rot) * sin.unsqueeze(unsqueeze_dim))

    q = torch.cat([q_rot, q_pass], dim=-1)
    k = torch.cat([k_rot, k_pass], dim=-1)
    return q, k
```

这一步非常值得先做，因为它改动小，但会让结构更接近 Qwen3.5。依据就是你文档里写的 **partial rotary factor = 0.25** 。

---

## 四、Gated DeltaNet 不要一上来就完整复刻

这是最难的部分。

你给的说明里，Qwen3.5 的线性注意力不是普通 linear attention，而是：

* Delta rule
* 门控
* 分组因果卷积
* 训练态 chunk 并行
* 推理态 recurrent O(1) cache 

这部分如果你直接硬上，基本等于重写半个模型。

### 更现实的做法

先分两个阶段：

#### 阶段 A：先用“伪线性 attention”占位

比如先实现一个简化版：

* Q/K/V 投影
* causal depthwise conv
* 简化状态更新
* 先不做严格的 recurrent delta rule

这样先把**混合层框架**搭起来。

#### 阶段 B：再替换成真正的 Gated DeltaNet

等你确认训练脚本、推理 cache、loss 都能工作后，再上：

* `chunk_gated_delta_rule`
* `recurrent_gated_delta_rule`
* conv state cache
* recurrent state cache

否则你会同时 debug：

* 新 attention
* 新 cache
* 新 norm
* 新 rope
* 新训练不稳定
  很容易炸。

---

## 五、你现在的 MoE 也不等于 Qwen3.5 MoE

你现在的 `MOEFeedForward` 是：

* 一个 router
* 多个 experts
* top-k 聚合 

但你文档里说的 Qwen3.5-MoE 是：

* **router experts**
* **shared expert**
* 最终输出 = 稀疏专家输出 + 共享专家输出 

所以如果你想做 **Qwen3.5-MoE 风格**，建议这样改：

```python
class Qwen35MoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([
            FeedForward(config, intermediate_size=config.moe_intermediate_size)
            for _ in range(config.num_experts)
        ])
        self.shared_expert = FeedForward(
            config, intermediate_size=config.shared_expert_intermediate_size
        )
        self.shared_gate = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
```

输出：

```python
sparse_out = ...
shared_out = self.shared_expert(x) * torch.sigmoid(self.shared_gate(x))
out = sparse_out + shared_out
```

这比你现在的 MoE 更像 Qwen3.5 的思路。

---

## 六、建议你的配置类这样扩展

你当前 `MiniMindConfig` 只有基础参数和简单 MoE 参数 。
建议至少加这些：

```python
self.attn_pattern = kwargs.get("attn_pattern", "linear,linear,linear,full")
self.partial_rotary_factor = kwargs.get("partial_rotary_factor", 0.25)
self.use_qk_norm = kwargs.get("use_qk_norm", True)
self.use_zero_centered_rmsnorm = kwargs.get("use_zero_centered_rmsnorm", True)

# gated attention
self.attn_gate = kwargs.get("attn_gate", True)

# linear attention / deltanet
self.linear_num_heads = kwargs.get("linear_num_heads", 16)
self.linear_num_kv_heads = kwargs.get("linear_num_kv_heads", 16)
self.linear_head_dim = kwargs.get("linear_head_dim", 128)
self.conv_kernel_size = kwargs.get("conv_kernel_size", 4)
self.use_deltanet = kwargs.get("use_deltanet", False)

# moe
self.use_shared_expert = kwargs.get("use_shared_expert", False)
self.shared_expert_intermediate_size = kwargs.get(
    "shared_expert_intermediate_size", self.intermediate_size
)
```

这样你后面做 ablation 会轻松很多。

---

## 七、最推荐的实际改造顺序

按工程可控性，我建议你这么干：

### 版本 1：Qwen3.5-attention-lite

先只做：

* ZeroCenteredRMSNorm
* partial RoPE
* gated attention
* block 可配置

这时每层仍然全是 full attention。
好处是：**基本不动训练和 cache 逻辑**。

### 版本 2：Hybrid stack

再做：

* `layer_types = [linear, linear, linear, full] * n`
* 线性层先用简化版实现

先保证能训、能推。

### 版本 3：真实 Gated DeltaNet

再把简化线性层替换为：

* grouped causal conv
* chunk / recurrent state
* cache

### 版本 4：Qwen3.5-MoE

最后再加：

* shared expert
* 更稳定 router 初始化
* top-k + shared expert 合并

---

## 八、一个你可以直接照着改的类结构

你现在最好把文件重构成这样：

```python
class MiniMindConfig(...)
class RMSNorm(...)
class ZeroCenteredRMSNorm(...)

class RotaryEmbedding(...)
def apply_partial_rotary_pos_emb(...)

class Qwen35GatedAttention(nn.Module): ...
class Qwen35GatedDeltaNet(nn.Module): ...

class FeedForward(nn.Module): ...
class Qwen35MoeBlock(nn.Module): ...

class MiniMindBlock(nn.Module):
    def __init__(self, layer_id, config):
        self.self_attn = build_attention(layer_id, config)
        self.input_layernorm = RMSNorm(...)
        self.post_attention_layernorm = RMSNorm(...)
        self.mlp = build_mlp(layer_id, config)

class MiniMindModel(nn.Module): ...
class MiniMindForCausalLM(...): ...
```

核心思想是：
**先把“单一 Attention 类”拆成“attention 工厂”**。
不然你后面一旦要混合层，就会越改越乱。

---

## 九、我对你这个项目的直白建议

如果你的目标是“让 minimind 看起来像 qwen3.5”，那就优先做这三件事：

1. **partial RoPE**
2. **gated full attention + zero-centered qk norm**
3. **3:1 hybrid layer schedule**

这三项改完，虽然还不是完整 Qwen3.5，但已经从结构上明显靠近了。
而真正最难的是 **Gated DeltaNet 的训练态 / 推理态双实现**，这个不要第一步就上。

---

## 十、给你一个结论

**不是“把 MiniMind 的模型换成 Qwen3.5”这么简单，而是要把当前统一 decoder block，改造成 Qwen3.5 的混合 block 框架。**
你现在的代码更接近“标准 RoPE + GQA + SwiGLU/MoE”的小型 decoder ；
而你提供的 Qwen3.5 说明对应的是“**Hybrid Attention + Gated Attention + Partial RoPE + 可选 MoE(shared expert)**”的体系 。

你要的话，我下一条可以直接给你一版 **`model_minimind.py` 的改造草图**，先帮你实现：

* `ZeroCenteredRMSNorm`
* `apply_partial_rotary_pos_emb`
* `Qwen35GatedAttention`
* `按 3:1 组装层`
