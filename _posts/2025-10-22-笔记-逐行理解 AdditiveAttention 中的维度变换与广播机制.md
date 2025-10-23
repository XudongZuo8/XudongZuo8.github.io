---
title: 逐行理解 AdditiveAttention 中的维度变换与广播机制  
date: 2025-10-23  
excerpt: 本文详细拆解了 PyTorch 中加性注意力（AdditiveAttention）的实现。我们将逐行跟踪张量（tensor）的维度变化，深入探讨 unsqueeze 和广播机制如何高效地计算所有 Query 和 Key 的配对，以及最终如何生成注意力分数。  
tags:

  -PyTorch
  -Attention  
  -深度学习  
  -NLP
---

在学习注意力机制时，`AdditiveAttention`（加性注意力）是一个经典实现。它出自《动手学深度学习》(d2l.ai) 课程，其代码实现非常精妙，尤其是 `forward` 函数中利用 PyTorch 广播（Broadcasting）机制来并行计算所有 Query 和 Key 的配对，堪称一行“神来之笔”。

然而，这行代码 `features = queries.unsqueeze(2) + keys.unsqueeze(1)` 对于初学者来说却极易造成困惑。它到底是如何工作的？张量的维度在每一步究竟发生了什么变化？

本文将基于我们此前的讨论，从头到尾、逐行逐个维度地为你彻底拆解这段代码，搞清从 QK 到最终注意力分数的每一步。

### 1. 目标代码：AdditiveAttention 类

首先，让我们贴出我们要分析的目标代码：

```python
#@save
class AdditiveAttention(nn.Module):
    """加性注意力"""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        # 步骤 1: 投影
        queries, keys = self.W_q(queries), self.W_k(keys)
        
        # 步骤 2: 广播与相加
        # queries的形状：(batch_size，查询的个数，1，num_hidden)
        # key的形状：(batch_size，1，“键－值”对的个数，num_hiddens)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        
        # 步骤 3: 计算分数
        # self.w_v仅有一个输出，因此从形状中移除最后那个维度。
        # scores的形状：(batch_size，查询的个数，“键-值”对的个数)
        scores = self.w_v(features).squeeze(-1)
        
        # 步骤 4: Softmax 归一化
        self.attention_weights = masked_softmax(scores, valid_lens)
        
        # 步骤 5: 加权求和
        # values的形状：(batch_size，“键－值”对的个数，值的维度)
        return torch.bmm(self.dropout(self.attention_weights), values)
```

### 2. 核心解惑：`features = queries.unsqueeze(2) + keys.unsqueeze(1)`

这是整个模块中最令人困惑的一步。我们来详细拆解它。

#### 步骤 2.1：`queries.unsqueeze(2)` 和 `keys.unsqueeze(1)`

这个操作的目的是**“创造”新的维度**，为广播做准备。

假设我们的参数如下（简化起见，`batch_size=1`, `num_hiddens=2`）：

- queries 形状为 (1, 2, 2)，代表有 2 个 query：

  $q_1 = [1, 2]$ 和 $q_2 = [3, 4]$。

- keys 形状为 (1, 3, 2)，代表有 3 个 key：

  $k_1 = [10, 20]$, $k_2 = [30, 40]$, $k_3 = [50, 60]$。

1. **`A = queries.unsqueeze(2)`**

   - 在第 2 索引（第 3 维）插入一个新维度。

   - 形状变化: (1, 2, 2) $\rightarrow$ (1, 2, 1, 2)

   - **值的布局**：

     ```python
     A = [  # Batch 0
           [  # num_queries = 2
             [[1, 2]],  # 这是 q1 (形状 [1, 2])
             [[3, 4]]   # 这是 q2 (形状 [1, 2])
           ]
         ]
     ```

2. **`B = keys.unsqueeze(1)`**

   - 在第 1 索引（第 2 维）插入一个新维度。

   - 形状变化: (1, 3, 2) $\rightarrow$ (1, 1, 3, 2)

   - **值的布局**：

     ```python
     B = [  # Batch 0
           [  # new_dim = 1
             [ [10, 20], [30, 40], [50, 60] ]  # 这是 k1, k2, k3 (形状 [3, 2])
           ]
         ]
     ```

#### 步骤 2.2：广播（Broadcasting）的魔法

现在我们执行 `features = A + B`。PyTorch 会比较 `A` 和 `B` 的形状：

- `A` 形状: `(1, 2, 1, 2)`
- `B` 形状: `(1, 1, 3, 2)`

广播机制会沿着大小为 1 的维度“拉伸”张量，使它们匹配：

- `A` 在 `dim 2` (key的维度) 上是 1，`B` 是 3。`A` 被拉伸 3 次。
- `B` 在 `dim 1` (query的维度) 上是 1，`A` 是 2。`B` 被拉伸 2 次。

相加时，内部的值变化如下：

```python
# A 被"拉伸"后的样子 (虚拟的)
[
  [
    [ [1, 2], [1, 2], [1, 2] ],  # q1 (为了匹配 k1, k2, k3)
    [ [3, 4], [3, 4], [3, 4] ]   # q2 (为了匹配 k1, k2, k3)
  ]
]

# B 被"拉伸"后的样子 (虚拟的)
[
  [
    [ [10, 20], [30, 40], [50, 60] ], # [k1, k2, k3] (为了匹配 q1)
    [ [10, 20], [30, 40], [50, 60] ]  # [k1, k2, k3] (为了匹配 q2)
  ]
]

# 逐元素相加后的 features (形状 [1, 2, 3, 2])
[
  [
    # --- 这是 q1 和所有 k 的组合 ---
    [ [11, 22],  # q1 + k1
      [31, 42],  # q1 + k2
      [51, 62] ],# q1 + k3

    # --- 这是 q2 和所有 k 的组合 ---
    [ [13, 24],  # q2 + k1
      [33, 44],  # q2 + k2
      [53, 64] ] # q2 + k3
  ]
]
```

**关键结论：** `features[b, i, j, :]` 这个向量，其**坐标 `(i, j)`** 完美地**对应**了“第 `i` 个 Query” 和 “第 `j` 个 Key” 的配对，而它**存储的值**就是 $q_i + k_j$ 的相加结果。

### 3. 从特征到分数：`scores = self.w_v(features).squeeze(-1)`

现在我们有了一个 `features` 张量，形状为 `(batch_size, num_queries, num_kv_pairs, num_hiddens)`，它存储了所有 QK 配对的组合特征。我们如何将其变为分数？

1. **`self.w_v(features)` (打分)**
   - `self.w_v` 是一个 `nn.Linear(num_hiddens, 1)`。
   - 它的作用就像一个**“评分裁判”**。它接收 `features` 最后一个维度的 `num_hiddens` 维向量（这个向量融合了** $q_i$ **和** $k_j$ **的信息），然后给这个配对打一个**标量分数**（一个单独的数字）。
   - 这个线性层作用在最后一个维度上，使形状从 `(..., num_hiddens)` 变为 `(..., 1)`。
   - **输出形状**：`(batch_size, num_queries, num_kv_pairs, 1)`。
2. **`.squeeze(-1)` (降维)**
   - 上一步我们得到了一个尾巴是 `1` 的形状，这个维度是多余的。
   - `squeeze(-1)` 的作用就是**移除最后一个维度**（前提是该维度的大小必须是1）。
   - **输出形状**：`scores` 的形状变为 `(batch_size, num_queries, num_kv_pairs)`。

### 4. 归一化：`masked_softmax(scores, valid_lens)`

我们拿到的 `scores` 是一个“原始分数”矩阵，`scores[b, i, j]` 代表 $q_i$ 对 $k_j$ 的原始好感度，它可以是任意实数（如 -20.5, 9.8）。

`masked_softmax` 的作用就是将这些原始分数转化为0到1之间的**“注意力权重”**。

- **归一化内容**：它归一化的**是每一个 `query`** 相对于**所有 `key`** 的**原始匹配分数**。
- **如何工作**：`softmax` 沿着**最后一个维度**（即 `num_kv_pairs` 维度）进行。
  - 例如，对于 `query` 0，它会取出 `scores[b, 0, :]` 这个分数向量（如 `[1.0, 4.0, 0.5]`）。
  - 然后计算 `softmax([1.0, 4.0, 0.5])`，得到 `[0.04, 0.90, 0.06]`。
  - `valid_lens` 的作用是在计算 `softmax` 之前，把那些“填充”的（无效的）`key` 对应的分数设置成一个非常小的负数（如 $-\infty$），这样它们在 `softmax` 后的权重就变成了 0。
- **输出**：`self.attention_weights`，形状为 `(batch_size, num_queries, num_kv_pairs)`。

### 5. 完整流程：使用测试数据追踪维度

最后，我们用一段示例代码来完整追踪一次维度的变化，这将澄清所有关于 `query_size`, `key_size` 和 `num_hiddens` 的困惑。

**测试数据：**

```python
queries = torch.normal(0, 1, size=(2, 1, 20))
keys = torch.ones((2, 10, 2))
values = torch.arange(40, ...).reshape(1, 10, 4).repeat(2, 1, 1)
valid_lens = torch.tensor([2, 6])
attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8, dropout=0.1)
```

**维度追踪：**

1. **初始输入**：
   - `queries`: `(2, 1, 20)` (batch=2, num_q=1, q_size=20)
   - `keys`: `(2, 10, 2)` (batch=2, num_kv=10, k_size=2)
   - `values`: `(2, 10, 4)` (batch=2, num_kv=10, v_size=4)
   - `num_hiddens = 8`
2. **`queries, keys = self.W_q(queries), self.W_k(keys)` (投影)**
   - `W_q` 是 Linear(20, 8) $\rightarrow$ queries 形状变为 `(2, 1, 8)`
   - `W_k` 是 Linear(2, 8) $\rightarrow$ keys 形状变为 `(2, 10, 8)`
   - **注意**：`query_size` 和 `key_size` 不同的问题在这里被解决了！它们都被投影到了相同的 `num_hiddens` 维度 (8)。
3. **`...unsqueeze(2) + ...unsqueeze(1)` (广播)**
   - `q_unsqueezed`: (2, 1, 8) $\rightarrow$ (2, 1, 1, 8)
   - `k_unsqueezed`: (2, 10, 8) $\rightarrow$ (2, 1, 10, 8)
   - 广播相加 `(2, 1, 1, 8) + (2, 1, 10, 8)`
   - `features` 形状: `(2, 1, 10, 8)`
4. **`scores = self.w_v(features).squeeze(-1)` (打分)**
   - `self.w_v` 是 `Linear(8, 1)` $\rightarrow$ `features` 形状变为 `(2, 1, 10, 1)`
   - .squeeze(-1) $\rightarrow$ scores` 形状变为 `(2, 1, 10)
5. **`masked_softmax(scores, valid_lens)` (归一化)**
   - `softmax` 沿着 `dim = -1` (即 10) 操作。
   - `attention_weights` 形状: `(2, 1, 10)`
6. **`torch.bmm(..., values)` (加权求和)**
   - 输入 1 (`attention_weights`): `(2, 1, 10)`
   - 输入 2 (`values`): `(2, 10, 4)`
   - 批量矩阵乘法 (`bmm`) 对每个批次执行：(1, 10)@ (10, 4) $\rightarrow$ (1, 4)
   - **最终输出形状**: `(2, 1, 4)`

### 总结

加性注意力的实现虽然简短，但它巧妙地融合了线性投影、张量维度扩展和广播机制，用完全并行的方式完成了所有 Query-Key 配对的计算。理解其核心——即**“投影（Projection）、升维（Unsqueeze）、广播（Broadcasting）、降维（Squeeze）”**这一系列组合操作——是掌握现代深度学习框架中张量操作的关键。