# 数学公式 → PyTorch 代码：翻译流程

> **这份记什么**：把论文/讲义里的 `∇θ J(θ)` 变成能跑的 `loss.backward()` 的**机械流程**。
> 不是 hw2 专属 —— hw3 (DQN/SAC)、hw4、final project (PPO/GRPO) 会反复用到同一套。
>
> **不记什么**：
> - 「为什么 PG 需要造一个假 loss」→ `NOTES.md` 阶段 1「关键决定」，那条讲决策和推论
> - 「跑出来多少」→ `REPORT.md`
>
> 分界线：**这里是流程，NOTES 是这次为什么这么决定**。

---

## 一句话总结

**你负责写 `∇` 里面的东西，PyTorch 负责补上那个 `∇`。**

```
∇J  ≈   ∇ [ (1/N) Σ log π(a|s) · A ]
        └┬┘ └──────────┬───────────┘
    autodiff 做      你写这坨（取负号后就是 loss）
```

loss **不是** `∇J`。loss 是 `∇` 作用的那个**对象**。

两个理由说明为什么不能拿 `∇J` 当 loss：

| # | 理由 |
|---|---|
| 1 | **维度对不上**。`∇J` 是向量（和 θ 同形，CartPole 默认 `-l 2 -s 64` 是 4610 个数），`.backward()` 只吃标量 |
| 2 | **鸡生蛋**。`∇J` 的表达式里还嵌着一个 `∇log π` —— 算它本来就得靠 autodiff，所以你拿不到「已经算好的 `∇J`」去喂给谁 |

---

## 五步翻译（以 PDF 式 (8) 为例）

起点：

```
∇θ J ≈ (1/N) Σ_{i=1..N} ( Σ_t ∇θ log π(a_t^i | s_t^i) ) ( Σ_t r(s_t^i, a_t^i) )
                          └────── 每步一个 ──────┘  └── 整条轨迹一个标量 ──┘
```

### 第 1 步：把「与求和变量无关的因子」推进求和号

`Σ_t r(s_t^i, a_t^i)` 是**整条轨迹的一个标量**，跟 `t` 无关，所以能塞进左边的 `Σ_t`：

```
= (1/N) Σ_i Σ_t  ∇θ log π(a_t^i | s_t^i) · R(τ^i)
                                            └ 对每个 t 都是同一个数
```

### 第 2 步：把它复制成「每个时间步一份」

既然每个 `t` 乘的都是同一个 `R(τ^i)`，就在每个时间步复制一份，记作 `A_t^i`。

> **这一步解释了 `_discounted_return` 里那个看起来很浪费的写法**：
> ```python
> return np.full(T, total, dtype=np.float32)   # 同一个标量复制 T 份
> ```
> 不是冗余 —— 是为了让式 (8) 能套进「每个时间步一行」的扁平布局。
>
> `-rtg` 时 `_discounted_reward_to_go` 给的 `A_t` 随 `t` 变化（式 11），
> **下游代码一个字都不用改**。所有变体的差异都被压进 `advantages` 这一个数组里了。

```
= (1/N) Σ_i Σ_t  ∇θ log π(a_t^i | s_t^i) · A_t^i
```

### 第 3 步：拍平双重下标

`(i, t)` 两层下标 → 一个扁平下标 `j`。这就是 `pg_agent.py` `update` 里那五行 `np.concatenate` 干的事：

```
= (1/N) Σ_{j=1..batch}  ∇θ log π(a_j | s_j) · A_j
```

> **前提**：`A_j` 必须在拍平**之前**按轨迹算好（折扣指数是轨迹内相对的）。
> 先拼接再算会静默算错，不报任何错。见 NOTES 阶段 1「flatten 必须在 `_calculate_q_vals` 之后」。

### 第 4 步：把 `∇` 提到外面

**成立的前提：`A_j` 与 θ 无关。** 它是从已采样数据算出的固定数字；代码里它走
`ptu.from_numpy` 进来，`requires_grad=False`，计算图天然是断的。

若 `A` 依赖 θ，则 `∇(log π · A) = ∇log π · A + log π · ∇A`，多出来的第二项会让整个推导失效。

```
= ∇ [ (1/N) Σ_j log π(a_j | s_j) · A_j ]
```

### 第 5 步：取负号，得到 loss

`optimizer.step()` 做的是 `θ ← θ - α·θ.grad`（**下降**），而我们要**上升**。
减号改不了，只能把 loss 取反：

```python
loss = -(log_probs * advantages).mean()
```

---

## 符号对照表

| 数学符号 | 代码 | 形状 |
|---|---|---|
| `π_θ(·\|s_j)` | `self(obs)` | 分布对象，`batch_shape=(batch,)` |
| `log π_θ(a_j\|s_j)` | `.log_prob(actions)` | `(batch,)`（连续需 `.sum(-1)`） |
| `A_j` | `advantages` | `(batch,)` |
| `(1/N) Σ_j` | `.mean()` | 标量 |
| `-`（因为要上升） | 最前面的负号 | |
| `∇θ` | `loss.backward()` | — |

> **一处没照抄公式**：式 (8) 除的是 `N`（轨迹条数），`.mean()` 除的是 `batch_size`（时间步总数）。
> 差一个「平均轨迹长度」的常数因子，被 learning rate 吸收了。标准做法，但要知道自己偏了。
> CartPole 里这个因子随策略变好而变大（轨迹变长），相当于学习率在隐式衰减。

---

## 落地代码（`MLPPolicyPG.update`）

```python
action_distribution = self(obs)
log_probs = action_distribution.log_prob(actions)
if not self.discrete:
    log_probs = log_probs.sum(dim=-1)        # (batch, ac_dim) -> (batch,)

loss = -(log_probs * advantages).mean()

self.optimizer.zero_grad()
loss.backward()
self.optimizer.step()
```

### `backward` 和 `optimizer` 是解耦的

这是最容易记混的机制。两者**不直接通信**，只通过参数上的 `.grad` 属性间接传递：

```
loss.backward()   →  给每个 p in parameters() 填上 p.grad     （optimizer 完全不知情）
optimizer.step()  →  读 p.grad，执行 p ← p - α·p.grad         （不知道 loss 是什么）
```

所以：

- **必须 `zero_grad()`** —— `.grad` 默认是**累加**（`+=`），上一轮的残留会污染这一轮
- **optimizer 只认它构造时收到的那批参数** —— `policies.py` `__init__` 里用
  `itertools.chain([self.logstd], self.mean_net.parameters())` 收参数，
  漏掉 `logstd` 的话它永远不更新，**不报错，只是训练上不去**

---

## 形状检查清单

写完先过一遍，这几处错了**都不报错，只是静默训练失败**：

| 检查 | 离散 (CartPole) | 连续 (HalfCheetah) |
|---|---|---|
| `self(obs)` 的 `batch_shape` | `(batch,)` | `(batch, ac_dim)` |
| `.log_prob(actions)` | `(batch,)` | `(batch, ac_dim)` ← **必须 `.sum(-1)`** |
| `log_probs * advantages` | `(batch,)` | `(batch,)` |
| `loss` | 0 维标量 | 0 维标量 |

**不能无脑 `.sum(-1)`**：离散情况下 `log_prob` 已经是 `(batch,)`，再 sum 会塌成标量，
整个 batch 的梯度糊成一团。必须 `if not self.discrete` 分支。

**命名坑**：`distributions`（`from torch import distributions` 的模块）
和 `distribution`（你的局部变量）只差一个字母。统一用 `action_distribution`，
和 `get_action` 里的命名保持一致。

**不需要 `.detach()`**：`advantages` 走 `ptu.from_numpy` 进来，`requires_grad=False`。
这正是第 4 步「`A` 与 θ 无关」在代码层面的保证。

---

## 判断题：这个 loss 是真的还是造的？

翻译之前先问一句，决定了你之后怎么 debug：

| | 真 loss | 代理 loss (surrogate) |
|---|---|---|
| 例子 | hw1 行为克隆的 MSE、hw2 `ValueCritic` 的回归损失 | hw2 `MLPPolicyPG` 的 actor loss |
| 推导方向 | 目标 → 求导 → 梯度 | **梯度 → 反推 → 一个假目标** |
| 数值有意义吗 | 有，该降，能拿来 debug | **没有，涨跌都说明不了事** |
| 同一批数据能走几步 | 多步（hw2 critic 默认 5 步，`-bgs`） | **1 步**，然后必须重新采样 |

最后一行的原因：`∇L = -∇J` **只在当前 θ、当前这批数据下成立**。数据是用旧的 `π_θ` 采的，
θ 一动等式就失效。想在一批数据上多走几步又不出错，就得加重要性采样修正 —— 那就是 PPO。

这个不对称在 `pg_agent.py` `update` 里看得很清楚：actor 走一次，critic 走
`self.baseline_gradient_steps` 次。

---

## 复用检查表

下次拿到一个 `∇θ J` 的公式，按顺序问：

- [ ] 这是**真 loss** 还是要**反推代理 loss**？（决定了数值有没有意义、能不能多步）
- [ ] 公式里哪些因子**与 θ 无关**？（决定 `∇` 能不能提出来；对应代码里哪些量要 detach / 走 numpy）
- [ ] 有没有「与求和变量无关、可以推进求和号」的因子？（第 1 步）
- [ ] 双重下标 `(i, t)` 怎么拍平？拍平**之前**必须算完哪些量？（第 2–3 步）
- [ ] 要上升还是下降？（负号）
- [ ] `log_prob` 的形状对不对？该不该 `.sum(-1)`？
- [ ] optimizer 构造时把所有可训练参数都收进去了吗？
