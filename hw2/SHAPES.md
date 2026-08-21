# 张量形状与轴：怎么想才不会错

> **这份记什么**：形状推理的心法 —— 哪根轴装什么、`dim=` 到底在说什么、
> 分布对象的 `batch_shape` / `event_shape`、哪些形状错误**不会报错**，
> 以及六个主流库实测下来怎么防（T0/T1/T1.5/T2）与**本仓库采用的约定**。
>
> **不记什么**：
> - 「公式怎么翻译成代码」→ `FORMULA_TO_CODE.md`
> - 「hw2 这次为什么这么决定」→ `NOTES.md`
> - 「跑出来多少」→ `REPORT.md`
>
> 和 `FORMULA_TO_CODE.md` 一样，这是**跨作业复用**的：hw3/hw4/hw5、以及之后写
> attention / LLM 训练时是同一套推理方式。

---

## 心法一：`dim=` 指的是「被**消掉**的那根轴」

`sum` / `mean` / `max` / `softmax` 这些归约操作，参数 `dim` 说的都是**哪根轴会消失**。

```python
t.shape == (3, 2)
t.sum(dim=-1).shape == (3,)     # 消掉轴 -1（长度 2 那根）
t.sum(dim=0).shape  == (2,)     # 消掉轴 0（长度 3 那根）
```

> 例外：`softmax(dim=-1)` 形状不变 —— 它是「沿这根轴归一化」而不是归约。
> 但「dim 指的是被作用的那根轴」这个直觉是一致的。

**所以问题永远转化成：我想合并的东西装在哪根轴上？**

---

## 心法二：`-1` = 特征维 / 事件维

深度学习里最后一根轴几乎总是「一个样本的内部结构」，前面的轴是「有多少个样本」。

```
(batch, feature)                 ← 最后一根是 feature
(batch, seq_len, hidden)         ← 最后一根是 hidden
(batch, ac_dim)                  ← 最后一根是动作的各个维度
```

写 `-1` 而不是写死的数字，是为了**对前面加几层 batch 不敏感**：

```
形状 (3, 2)      sum(dim=-1) -> (3,)       sum(dim=1) -> (3,)      ← 恰好一样
形状 (3, 4, 2)   sum(dim=-1) -> (3, 4)     sum(dim=1) -> (3, 2)    ← 不一样了
```

同一条规则之后会在这些地方复用：`softmax(dim=-1)`（attention 对 key 归一化）、
`logsumexp(dim=-1)`、`norm(dim=-1)`、`cross_entropy` 内部对 vocab 维的归约。

---

## 分布对象的两个形状

这是 `torch.distributions` 最容易搞错的地方，也是 hw2 里 `if not self.discrete`
分支存在的**唯一**原因。

| | 含义 |
|---|---|
| `batch_shape` | 有**多少个互相独立**的分布 |
| `event_shape` | **一次抽样**产生的东西是什么形状 |

`log_prob(x)` 的输出形状 = `batch_shape`（`event_shape` 那部分被「消化」掉了）。

### 本作业实测

```
离散  Categorical(logits=(5,2))          batch_shape=[5]    event_shape=[]
                                          log_prob((5,))   -> (5,)

连续  Normal(loc=(5,6), scale=(6,))       batch_shape=[5,6] event_shape=[]
                                          log_prob((5,6))  -> (5,6)     ← 注意！
```

**关键：`Normal` 的 `batch_shape` 是 `[5, 6]`。**

PyTorch 把它理解成「**30 个互相独立的一维高斯**」，它**不知道**那 6 个数属于同一个动作。
`event_shape=[]` 就是在说「一次抽样只产生一个标量」。

所以 `log_prob` 给的是 6 个**边缘**概率，不是这个动作的**联合**概率。

### 为什么离散不需要合成

离散动作**本来就只占一个格子** —— 一个整数就是完整的动作。
`Categorical` 的 `batch_shape=[5]`、`event_shape=[]`，`log_prob` 直接给的就是联合概率，
没有东西需要合成。

**对称的说法**：两边都要联合概率，只是离散天然就是，连续要自己乘起来。

### 合成 = 相乘 = log 域相加

各维度独立，所以

```
π(a|s) = π(a₁|s) · π(a₂|s) ··· π(a₆|s)
log π(a|s) = Σₖ log π(aₖ|s)
```

要合的是「同一个动作的各个关节」，它们装在**最后一根轴**上 → `.sum(dim=-1)`。

```python
log_probs = action_distribution.log_prob(actions)
if not self.discrete:
    log_probs = log_probs.sum(dim=-1)      # (batch, ac_dim) -> (batch,)
```

**不能无脑 `.sum(-1)`**：离散时 `log_prob` 已经是 `(batch,)`，再 sum 会塌成标量，
整个 batch 的梯度糊成一团。

### 加错轴会怎样

`batch=3, ac_dim=2`：

```
              关节0   关节1
step 0     [ -1.0   -2.0 ]
step 1     [ -0.5   -1.5 ]
step 2     [ -3.0   -1.0 ]
```

| 写法 | 方向 | 结果 | 形状 | |
|---|---|---|---|---|
| `.sum(dim=-1)` | 按行 | `[-3.0, -2.0, -4.0]` | `(3,)` | ✅ 每步一个动作的联合概率 |
| `.sum(dim=0)` | 按列 | `[-4.5, -4.5]` | `(2,)` | ❌ 把不同时间步的动作混在一起 |
| `.sum()` | 全加 | `-9.0` | 标量 | ❌ 整个 batch 塌成一个数 |

`sum(dim=0)` 算的是「关节 0 在所有时间步的 log 概率之和」—— 没有任何意义。
不同时间步是不同状态下的不同动作，它们的概率不该相乘。

### `Independent`：把 `.sum(-1)` 封进分布对象

```python
D.Independent(D.Normal(loc=mean, scale=std), 1)
#  batch_shape=[5]   event_shape=[6]     ← 6 从 batch 挪进了 event
#  log_prob((5,6)) -> (5,)               ← 直接是联合概率
```

实测与手动 `.sum(-1)` 数值完全一致。参数 `1` 的意思是「把**最后 1 个**维度
从 batch 重新解释为 event」。

用了它，`if not self.discrete` 分支可以整个删掉。**但手动 sum 让「为什么要合成」显式可见**，
学习阶段更推荐手动写。

---

## 本作业四个环境的真实维度

`ac_dim` 在两种动作空间下**含义完全不同**，这是所有形状混乱的源头：

| 环境 | ob_dim | 动作空间 | ac_dim 的含义 |
|---|---|---|---|
| CartPole-v0 | 4 | discrete | **2** = 候选动作的**个数** |
| LunarLander-v2 | 8 | discrete | **4** = 候选动作的**个数** |
| InvertedPendulum-v4 | 4 | continuous | **1** = 动作向量的**维度** |
| HalfCheetah-v4 | 17 | continuous | **6** = 动作向量的**维度** |

---

## 失败模式：哪些会报错，哪些静默

**这一节是本文档最有用的部分。** 实测「忘了 `.sum(-1)`」的后果：

| 场景 | `log_probs * advantages` | 后果 |
|---|---|---|
| `batch=1000, ac_dim=6`（HalfCheetah） | RuntimeError | ✅ **会崩，能发现** |
| `batch=1000, ac_dim=1`（**InvertedPendulum**） | 广播成 `(1000, 1000)` | ⚠️ **静默算错** |
| `batch=6, ac_dim=6`（碰巧等长） | 广播成 `(6, 6)` | ⚠️ **静默算错** |

**`ac_dim=1` 是最危险的**：任何 `(batch, 1)` 都能和 `(batch,)` 广播成 `(batch, batch)`，
永远不报错。实验 4 的 InvertedPendulum 正好是 `ac_dim=1`。

> 广播规则：从**右往左**对齐，长度相等或其中一个为 1 就能广播。
> `(1000, 6)` vs `(1000,)` → 右对齐是 `6` vs `1000` → 不匹配 → 报错。
> `(1000, 1)` vs `(1000,)` → 右对齐是 `1` vs `1000` → **1 可以广播** → 静默展开成 `(1000,1000)`。

### 通用防御

在关键边界写死断言，比事后 debug 便宜得多：

```python
assert log_probs.shape == advantages.shape, (log_probs.shape, advantages.shape)
```

`pg_agent.py` 的 `_estimate_advantage` 里 starter code 已经放了一条同类的：
`assert values.shape == q_values.shape`（网络输出是 `(batch,1)`，会撞上）。

---

## 验证用例的设计原则

**永远不要用「各维长度相等」或「含 1」的形状做验证。** 它们会让错误的实现蒙混过关：

- `batch == ac_dim` → 加错轴也能广播成功
- `ac_dim == 1` → 任何广播都成功
- 等长轨迹 → 拍平前后算出的 Q 值可能碰巧一样（NOTES 阶段 1「坑 1」踩过这个）

好的验证形状：**各维长度互不相等、都大于 1**。比如 `batch=7, ac_dim=3`。

---

## 怎么系统性地防：工业界实测

前面讲的是**广播规则**和**失败模式**。这一节讲**怎么让形状从「隐含假设」变成「写出来的东西」**。

先说一个坏消息：**没有更安全的 API**。三种常见去轴写法，形状不符时全部静默通过：

```
(3,5).squeeze(-1)   ->  (3,5)    什么都没做，不报错
(3,5)[:, 0]         ->  (3,)     取了第 0 列，不报错
(3,5).reshape(-1)   ->  (15,)    拍平了，不报错
```

额外两个坑：`squeeze(dim)` 在该维不是 1 时**静默无操作**；`squeeze()` 不带参数在 batch=1 时把 batch 维也吃掉（`(1,1) -> ()`）。

所以问题不是「选哪个函数」，而是「用什么机制把形状显式化」。四种机制：

| 档 | 机制 | 形状信息存在哪 |
|---|---|---|
| **T0** | docstring / 行内注释写形状 | 注释里 |
| **T1** | `assert` / `raise` 检查形状 | 运行时 |
| **T1.5** | 变量名后缀（Noam 记法） | 代码本身，**每一处使用** |
| **T2** | 类型注解（jaxtyping + beartype） | 类型系统，**只在函数边界** |

### 六个主流库的实测（2026-08 读的源码）

| 库 | 归属 | T0 | T1 | T1.5 | T2 |
|---|---|---|---|---|---|
| **verl** | 字节，当前 RLHF 事实主流 | ✅ 大量 | ❌ | ❌ | ❌ |
| **torchrl** | PyTorch 官方 | ✅ | ✅ `raise RuntimeError` | ❌ | ❌ |
| **TRL** | HuggingFace | ✅ 行内 `(B, L-1, H)` | ❌ | 少量 | ❌ |
| **RLinf** | 清华系 RL 基础设施 | ✅ `algorithms/` | ✅ `assert` | ❌ | 装了 einops 但算法里没用 |
| **torchtitan** | Meta 官方大规模训练 | ❌ | ❌ | ❌ | ❌ |
| **nanodo** | Google DeepMind | ❌ | ❌ | ✅ **全面 Noam 记法** | ❌ |

**六个库，零个用 T2。**

具体证据：

```python
# verl —— 纯 T0
"""shape is (bs, response_length)"""
"""shape: (bs, response_length, vocab_size)"""

# torchrl —— T0 + 真 T1（raise 不是 assert）
"""All tensors must have shape [*Batch x TimeSteps x *F]"""
if not (next_state_value.shape == state_value.shape == reward.shape
        == done.shape == terminated.shape):
    raise RuntimeError(SHAPE_ERR)

# nanodo —— 教科书级 T1.5
y_BxLxD = self.embed(y_BxL)
q_BxLxHxDh, k_BxLxHxDh, v_BxLxHxDh = ...
att_BxHxLxL = jnp.einsum(...)
logits_BxLxV = self.embed.attend(...)

class DoConfig:
  D: int  # model/embed dim
  H: int  # num attention heads
  L: int  # max context length
  V: int  # vocab size

# torchtitan —— 什么都没有
def forward(self, x: torch.Tensor) -> torch.Tensor:
```

### 两条分野

**① PyTorch 生态 → T0；JAX/Google 生态 → T1.5**

一个合理解释：**JAX 代码大量用 `einsum`，而 einsum 的字符串本身就是形状标注** ——
`jnp.einsum('BLD,DHK->BLHK', inputs_BLD, w_q_DHK)` 里字符串和后缀天然对齐、互相校验。
PyTorch 更多用 `.view()/.transpose()`，没有这种对应，后缀就成了纯额外负担。

**② 库的定位决定要不要 T1**

torchrl 是**给外部用户的库**，形状错了必须给好报错 → 上 T1。
verl / torchtitan 是**给专家读的实现**，假定使用者知道形状 → 不上。

### 关于 T2 和 CS336

CS336 要求 jaxtyping 是**教学选择**，不是行业实践 —— 那门课整节在做
`(B,L,D) -> (B,L,H,K)` 的变换，把形状放进类型系统能让学生的错误第一时间报出来。
**生产代码假定你已经会了**。

同理 torchtitan 的「什么都不写」不是懒 —— 它是 reference implementation，
目标是代码尽可能短、让人看清算法骨架，读者预设是 PyTorch 熟手。

### 写 vs 读，要求不一样

| | 要求 | 理由 |
|---|---|---|
| **T0** | **必须会写** | 六个库里五个在用，事实标准，成本一行 |
| **T1.5** | **必须会读**，写不写随意 | 迟早要读 Gemma / MaxText / nanodo 血统的代码，不认识 `q_BxLxHxDh` 会读不懂 |
| **T1** | **按场景** | 写库给别人用 → 上；写实验脚本 → 不必 |
| **T2** | **知道存在即可** | 零个生产库在用 |

> 最值钱的其实不在这四档里：是**读懂广播规则本身**（右对齐、左补 1）。
> 上面所有机制都只是提醒，真正让你不出错的是知道 `(B,1)` 和 `(B,)` 相乘会发生什么。

---

## 本仓库采用的约定：T0 + T1 + T1.5

学习阶段三种一起上 —— 生产代码假定你已经会了，而我们正在「会」的路上。

### 字母表

| 字母 | 含义 | 典型值 |
|---|---|---|
| `B` | batch（拍平后的时间步总数） | 1000 / 4000 / 5000 |
| `O` | ob_dim（观测维度） | CartPole 4、HalfCheetah 17 |
| `A` | ac_dim（**连续**动作的向量维度） | HalfCheetah 6、InvertedPendulum 1 |
| `T` | 单条轨迹的长度 | 变长 |
| `N` | 一个 batch 里的轨迹条数 | 变长 |

> 离散动作**没有** `A` —— 一个动作就是一个整数，`actions` 是 `(B,)` 不是 `(B,A)`。
> 这个不对称正是 `ac_dim` 两种含义的根源（见「本作业四个环境的真实维度」）。

### 本作业主要张量的后缀

| 变量 | 形状 | 后缀 |
|---|---|---|
| `obs` | `(B, O)` | `obs_BO` |
| `actions`（离散） | `(B,)` | `actions_B` |
| `actions`（连续） | `(B, A)` | `actions_BA` |
| `q_values` / `advantages` / `values` | `(B,)` | `..._B` |
| `log_prob` 求和**前**（连续） | `(B, A)` | `log_probs_BA` |
| `log_prob` 求和**后** | `(B,)` | `log_probs_B` |

**最后两行是这套记法的价值所在**：

```python
log_probs_BA = action_distribution.log_prob(actions_BA)   # (B, A)
log_probs_B  = log_probs_BA.sum(dim=-1)                   # (B,)
loss = -(log_probs_B * advantages_B).mean()               # 后缀对上了
```

如果漏了 `.sum(-1)`，代码会长成 `-(log_probs_BA * advantages_B)` ——
`BA` 乘 `B` 一眼就不对。**这正是 `SHAPES.md` 开头那个静默广播 bug**，
用后缀写出来就变成了可见的。

### 三档各用在哪

```python
def forward(self, obs_BO: torch.Tensor) -> torch.Tensor:
    """(B, O) -> (B,)                                    # T0：契约写出来
    network 的 output_size=1，输出 (B,1)，squeeze 掉那根记账用的轴。
    """
    values_B1 = self.network(obs_BO)
    assert values_B1.shape == (obs_BO.shape[0], 1), values_B1.shape   # T1：边界钉住
    return values_B1.squeeze(-1)                          # T1.5：名字说清形状
```

- **T0** 放在**函数 docstring** —— 每个有形状变换的函数一行
- **T1** 放在**边界**：函数入口、两个张量首次相遇处。starter code 自带的
  `assert values.shape == q_values.shape`（`pg_agent.py:177`）就是这一档，**别删**
- **T1.5** 用在**形状会变的局部变量**上。`x`、`i`、`loss` 这种不用加

**不要三档都堆在每一行** —— 那样噪声盖过信号。判据：
**这个量的形状会不会被搞错？** 会 → 上；不会 → 省掉。

## 复用检查表

拿到一个形状不确定的张量，按顺序问：

- [ ] 每根轴装的是什么？哪根是「样本」，哪根是「样本内部结构」？
- [ ] 我要合并的东西在哪根轴上？（`dim=` 填的就是它）
- [ ] 归约之后的形状，和下游要的形状对得上吗？
- [ ] 如果对不上，是会**报错**还是会**广播**？（含 1 的维度一律怀疑）
- [ ] 分布对象：`batch_shape` 和 `event_shape` 各是什么？`log_prob` 输出 = `batch_shape`
- [ ] 验证用例的各维长度互不相等吗？
