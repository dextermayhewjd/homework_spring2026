# HW2 实现笔记

> **这份记什么**：我怎么走到最终代码的 —— 写错的版本、什么现象让我发现错了、试过又放弃的写法。
>
> **不记什么**：
> - 「最终代码为什么长这样」→ 写进源文件 docstring，沿用 hw1 的三段式（*做的事 / 为什么 / 术语解释*）
> - 「跑了什么、结果多少」→ 写进 `REPORT.md`
>
> 分界线：**读最终代码的人需要知道的** 留在代码里；**只有写代码的我经历过的** 留在这里。

---

## 阶段总览

| 阶段 | PDF 节 | 动的文件 | commit | 状态 |
|---|---|---|---|---|
| 0. 数据流打通 | —（前置） | `run.py` `utils.py` `policies.py` | `b14695f` | ✅ |
| 1. Vanilla PG | §3 | `pg_agent.py` `policies.py` | | 🔄 6/8 |
| 2. Baseline   | §4 | `critics.py` `pg_agent.py` | | ☐ |
| 3. GAE        | §5 | `pg_agent.py` | | ☐ |
| 4. 调参       | §6 | 无代码改动 | — | ☐ |

> 三个阶段动的代码不重叠，**一个阶段一个 commit**。这样 `git log -p` 本身就是一份
> 精确到行的实现过程记录，零额外成本。写完一个阶段跑一次 `smart-commit`。

---

## 实现策略：为什么中途换了方向

前后两段用的是**相反**的推进方式。不是随意换的，是因为待解决问题的性质变了。

**阶段 0 —— 自顶向下，沿调用链**
`run.py` 的训练循环 → `utils.sample_trajectories` → `policy.get_action`，顺着调用往下追，
遇到 `None` 就填。因为那批 TODO 全是「数据怎么流动」的问题：必须先知道 `run.py` 要什么，
才知道 `utils` 该返回什么形状。自底向上根本无从下手。

**阶段 1–3 —— 自底向上，沿依赖链**
数据流在阶段 0 已经定死了（`agent.update` 的签名固定，进来就是 `list[array]`）。
剩下的全是数学，依赖关系是纯计算依赖：

```
_discounted_return / _discounted_reward_to_go   ← 无依赖，纯函数，可手算验证
        ↓
_calculate_q_vals                                ← 只依赖上面两个
        ↓
update 里的 flatten                              ← 依赖 q_values 的结构
        ↓
_estimate_advantage                              ← 依赖 flatten 后的形状
        ↓
MLPPolicyPG.update                               ← 依赖 advantages
```

若继续自顶向下，就会从 `PGAgent.update` 开写，但它调的每个东西都返回 `None` ——
跑不起来、验证不了，只能一路写到底再一次性 debug 十几个交织在一起的形状错误。
自底向上则是每写完一个就能单独验证。

> **判据**：TODO 在问「数据长什么样」→ 自顶向下；在问「这个数怎么算」→ 自底向上。
> hw3/hw4/hw5 的起始代码是同一套骨架，同样适用。

---

## 阶段 0 — 数据流打通（commit `b14695f`）

不对应 PDF 的任何实验节，是能跑起来的前置。8 个 TODO：

| # | 位置 | 写了什么 |
|---|---|---|
| 1 | `run.py:62` | `utils.sample_trajectories(...)` |
| 2 | `run.py:76` | `agent.update(...)` |
| 3 | `utils.py:32` | `policy.get_action(obs=ob)` |
| 4 | `utils.py:35` | `env.step(action=ac)` |
| 5 | `utils.py:38` | `steps >= max_length or done` |
| 6 | `policies.py:117` | `get_action`：from_numpy → `self(obs)` → `.sample()` → to_numpy |
| 7 | `policies.py:131` | `forward` 离散：`Categorical(logits=...)` |
| 8 | `policies.py:136` | `forward` 连续：`Normal(loc=mean, scale=exp(logstd))` |

> 表内顺序是按调用链重建的，git 只能证明这 8 个在同一个 commit 里，证明不了先后。

### 关键决定

概念性的理由已经写进 `policies.py` 的模块 docstring（为什么 forward 返回分布对象、
logstd 为什么存 log、itertools.chain 收参数、exp 与 softmax 的分工）。这里只补过程性的。

> TODO：`run.py:72` 那行字典推导式（N 个 dict → 一个 dict、值是 N 个 array 的 list）
> 当时是读懂了还是滑过去了？它决定了 `agent.update` 收到的是**保留轨迹边界**的 list，
> 直接决定阶段 1 为什么必须按轨迹逐条算 Q 值 —— 先拼接再算会静默算错。

### 踩的坑

> TODO

### 遗留问题

- [ ] `run.py`、`utils.py`、`policies.py` 里共有 8 条已完成但没删的 stale TODO 注释。
      之后 `grep -rn TODO` 数剩余工作量时会被干扰（24 条里只有 16 条是真活）

---

## 阶段 1 — Vanilla Policy Gradient（PDF §3）

跑通目标：CartPole 能收敛到 200。此阶段**不碰** `critics.py`，也不碰 GAE
（PDF §3.1 原文允许跳过 `use_baseline=True` 才走到的分支）。

### TODO 清单

> 行号会随着你写而漂移，按函数名找。

- [x] `pg_agent.py` `_discounted_return` —— PDF 式 (25) ✅ 已验证
- [x] `pg_agent.py` `_discounted_reward_to_go` —— PDF 式 (26) ✅ 已验证
- [x] `pg_agent.py` `_calculate_q_vals` 两个分支 ✅ 已验证
- [x] `pg_agent.py` `update` step 1.5：flatten ✅ 已验证
- [x] `pg_agent.py` `_estimate_advantage`：`critic is None` 分支 ✅ 已验证
- [x] `pg_agent.py` `_estimate_advantage`：advantage 归一化 ✅ 已验证
- [ ] `pg_agent.py` `update` step 3：调用 actor.update
- [ ] `policies.py` `MLPPolicyPG.update`：loss + optimizer step

### 关键决定

**`_discounted_reward_to_go` 用倒序递推而不是正序双重循环**

正序写法对每个 t 重扫一遍后缀，是 O(H²)。利用恒等式

```
G_t = Σ_{t'=t} γ^(t'-t) r_t'
    = r_t + γ · Σ_{t'=t+1} γ^(t'-(t+1)) r_t'
    = r_t + γ · G_{t+1}
```

倒着扫一遍即可，O(H)。CartPole 一条轨迹 200 步差别不大，
但 LunarLander 的 `--ep_len 1000` 是 100 万次 vs 1000 次。

这个恒等式成立**只因为指数是相对的 `t'-t`**。`_discounted_return` 的指数是绝对的 `t'`
（PDF 式 25），没有这个递推结构 —— 但它只算一个标量，本来就是 O(H)，不需要。
两个函数写法不同，根源就在这里。

**`running = 0.0` 免掉边界特判**：初始值等价于虚拟的 `G_T = 0`，
所以第一轮 `t = T-1` 时 `running = r[T-1]`，正好是边界条件。
阶段 3 写 GAE 时会再用一次同一个技巧（对应 `values` 后面 append 的那个 0）。

**返回 `np.ndarray` 而不是 list**：下游 `update` 里要 `np.concatenate`，
直接给 ndarray 省一次转换。dtype 不用纠结，`ptu.from_numpy` 里有 `.float()` 会统一。

**flatten 必须在 `_calculate_q_vals` 之后**：Q 值要按轨迹逐条算（折扣指数是轨迹内的），
一旦 `rewards` 被 `np.concatenate` 拍平，轨迹边界就丢了。
验证方式：3 条长度 `[4,2,3]` 的轨迹、γ=1、reward 全 1，
正确结果是 `[4,3,2,1, 2,1, 3,2,1]`（每条各自重新数）；
若顺序颠倒会得到 `[9,8,7,...,1]`。

**拍平抹掉的只是「数组布局上的边界」**，语义边界被搬进了 `terminals`
（`terminals[i]==1` 表示第 i 行是所在轨迹最后一步）。
阶段 1/2 不需要它 —— 非 GAE 分支里 `rewards` 和 `terminals` 两个参数压根没被引用；
阶段 3 的 GAE 递推 `A_t = δ_t + γλ·A_{t+1}` 要看下一时间步，就必须靠 `terminals` 断开。

**无 baseline 时 `advantages = q_values`**，不是「减去一个常数 b」。
本作业的 baseline 特指 §4 学出来的 `V_φ(s)`，`use_baseline=False` 就是它不存在。
不过常数基线的效果确实存在 —— 在 `-na` 里：`(A-μ)/(σ+ε)` 减掉的 μ 就是一个常数基线。
这正是 PDF §4.2 说「advantage normalization 让简单环境不再需要 baseline」的由来。
差别：减真常数 b 无偏；减 batch 均值 μ 技术上有偏（μ 依赖采样到的这批数据，PDF §2.3）。

### 验证记录

用 AST 从文件里抽出这两个方法单独执行（不需要装 torch），四组用例全过：

| 用例 | `_discounted_return` | `_discounted_reward_to_go` |
|---|---|---|
| γ=0.5, `[1,1,1]` | `[1.75]×3` | `[1.75, 1.5, 1.0]` |
| γ=1.0, `[1,2,3,4]` | `[10]×4` | `[10, 9, 7, 4]` |
| γ=0.9, `[5]`（长度 1） | `[5.0]` | `[5.0]` |
| γ=0.99, `[0,0,1,0]` | `[0.9801]×4` | `[0.9801, 0.99, 1.0, 0.0]` |

最后一组是最有鉴别力的：稀疏奖励能验出「指数是相对还是绝对」。
`[1,1,1]` 那组反而验不出来 —— 写错成绝对指数也能蒙对。
长度为 1 那组验证了 `running = 0.0` 的边界处理。

`_calculate_q_vals` / flatten / 归一化的验证（同样用 AST 抽方法执行，`_estimate_advantage`
和 actor 用桩替代）：

| 检查 | 结果 |
|---|---|
| 不等长轨迹 `[1,1,1]`/`[1,1]`，γ=1 | Case1 `[[3,3,3],[2,2]]`、Case2 `[[3,2,1],[2,1]]` ✅ |
| γ=0.5 同样两条 | 第二条 Case1 得 `1.5` 而非接着 γ³ 累加 → 轨迹间不串扰 ✅ |
| flatten 后形状（3 条 `[4,2,3]`，ob_dim=4） | `obs (9,4)`、其余 `(9,)` ✅ |
| flatten 后 q_values | `[4,3,2,1,2,1,3,2,1]` → 边界未串 ✅ |
| 归一化 `[1,2,3,4]` | `mean=0, std=1` ✅ |
| 归一化退化情形 `[7,7,7,7]`（std=0） | 得 `[0,0,0,0]`，未出现 NaN/inf ✅ |

### 踩的坑

#### 坑 1：把整个 `list[array]` 喂给了只吃单条轨迹的 helper

- **症状**：`_calculate_q_vals` 里写成 `self._discounted_return(rewards=rewards)`，
  用两条不等长轨迹 `[[1,1,1], [1,1]]` 一跑就崩：
  ```
  Case 1  ValueError: operands could not be broadcast together with shapes (3,) (2,)
  Case 2  ValueError: setting an array element with a sequence.
  ```
- **原因**：**类型标注差了一层嵌套**。
  ```python
  def _discounted_reward_to_go(self, rewards: Sequence[float])       # 一条轨迹
  def _calculate_q_vals(self,        rewards: Sequence[np.ndarray])  # 一堆轨迹
  ```
  helper 里 `T = len(rewards)` 算出来是**轨迹条数**而不是时间步数，
  `rewards[t]` 取出来是一整个 array 而不是一个 float，
  于是 `out[t] = running` 变成「把数组塞进标量位置」。
- **改法**：剥一层，用列表推导式逐条应用：
  ```python
  q_values = [self._discounted_return(single_reward_traj) for single_reward_traj in rewards]
  ```
- **教训**：
  1. 遇到形状类报错，**先对比上下游函数的类型标注**，`Sequence[float]` vs
     `Sequence[np.ndarray]` 这种一层之差最容易看漏。
  2. 这次运气好崩了。如果两条轨迹**碰巧等长**，Case 1 不会报错，而是静默算出
     折扣指数跨轨迹累加的错误结果 —— 所以验证用例必须用**不等长**轨迹。
  3. hw3 处理 batch 时大概率还会见到
     `setting an array element with a sequence`，症状→根因的诊断路径同样适用。

### 遗留疑问

> TODO：当时没搞懂、先绕过去的点

---

## 阶段 2 — Neural Network Baseline（PDF §4）

跑通目标：HalfCheetah `cheetah_baseline` 末尾 Eval return > 300。

### TODO 清单

- [ ] `critics.py` `ValueCritic.forward`
- [ ] `critics.py` `ValueCritic.update`：loss + optimizer step
- [ ] `pg_agent.py` `_estimate_advantage`：跑 critic 拿 values
- [ ] `pg_agent.py` `_estimate_advantage`：有 baseline、无 GAE 时的 advantages
- [ ] `pg_agent.py` `update` step 4：做 `baseline_gradient_steps` 次 critic 更新

### 关键决定

> TODO：critic 的训练目标用的是什么？为什么是它？

### 踩的坑

> TODO
>
> 已知会拦你的两处，撞到了就照模板记下来：
> - `_estimate_advantage` 里 `assert values.shape == q_values.shape`（网络输出是 `(batch,1)`）
> - values 忘了转 numpy 导致梯度串到 actor 上（这个不报错，只是训练变怪）

### 遗留疑问

> TODO

---

## 阶段 3 — Generalized Advantage Estimation（PDF §5）

跑通目标：LunarLander 五个 λ 里最好的一个，训练中至少出现一次 Eval return > 150。

### TODO 清单

- [ ] `pg_agent.py` `_estimate_advantage`：GAE 倒序递推

### 关键决定

> TODO：`terminals` 是怎么用来切断轨迹边界的？为什么要在 values 后面 append 一个 0？

### 踩的坑

> TODO

### 遗留疑问

> TODO：比如 PDF 式 (18) 有个归一化常数 1/(1-λ^(H-t-1))，式 (21) 的递推实现里去哪了？

---

## 阶段 4 — 调参（PDF §6）

无代码改动。调参过程记在 `REPORT.md` §4.1 的表里，不写在这。

---

## 跨作业复用

> 这节留到 hw2 全部做完再回填 —— 挑出对 hw3（DQN / SAC）、hw5 有用的结论。

> TODO
