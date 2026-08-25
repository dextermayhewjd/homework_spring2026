# HW2 实现笔记

> **这份记什么**：我怎么走到最终代码的 —— 写错的版本、什么现象让我发现错了、试过又放弃的写法。
>
> **不记什么**：
> - 「最终代码为什么长这样」→ 写进源文件 docstring，沿用 hw1 的三段式（*做的事 / 为什么 / 术语解释*）
> - 「跑了什么、结果多少」→ 写进 `REPORT.md`
> - 「数学公式怎么机械地翻译成 PyTorch 代码」→ 写进 `FORMULA_TO_CODE.md`，
>   那是跨作业复用的流程（hw3/hw4/final project 会再用），不该埋在 hw2 的某个阶段里
> - 「张量形状/轴怎么想、哪些形状错误不报错」→ 写进 `SHAPES.md`，同样是跨作业复用的
> - 「方差为什么是策略梯度的核心问题」→ 写进 `VARIANCE.md`，一个可枚举的小例子 +
>   本作业四种方差削减技巧的对照表，同样跨作业复用
> - 「因子多到跑不完全因子时，怎么系统地设计消融实验」→ 写进 `ABLATION.md`，
>   正交表的构造、平衡性为什么能分离效应、主效应怎么算怎么读，同样跨作业复用
>
> 分界线：**读最终代码的人需要知道的** 留在代码里；**只有写代码的我经历过的** 留在这里。

---

## 阶段总览

| 阶段 | PDF 节 | 动的文件 | commit | 状态 |
|---|---|---|---|---|
| 0. 数据流打通 | —（前置） | `run.py` `utils.py` `policies.py` | `b14695f` | ✅ |
| 1. Vanilla PG | §3 | `pg_agent.py` `policies.py` | `839e29c` `b049f09` `f762a8b` | ✅ 8/8 |
| 2. Baseline   | §4 | `critics.py` `pg_agent.py` | `3ca746e` `9cb3a21` `8422614` | ✅ 5/5 |
| 3. GAE        | §5 | `pg_agent.py` | 本次 | ✅ 1/1 |
| 4. 调参       | §6 | 无代码改动 | 本次（L9 部分） | 🔄 L9 已跑，确认实验/交互/因子7 未做 |

> 三个阶段动的代码不重叠，**一个阶段一个 commit**。这样 `git log -p` 本身就是一份
> 精确到行的实现过程记录，零额外成本。写完一个阶段跑一次 `smart-commit`。
>
> **实际执行时阶段 1 跨了三个 commit**（两个 estimator / 6-of-8 / 收尾+实验），
> 因为中途停下来补了理论推导和实验。约定没守住，但 `git log -p -- hw2/src` 仍然是完整记录，
> 不改历史。阶段 2、3 争取守住。

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

- [ ] 已完成但没删的 stale TODO 注释。截至阶段 1 结束重新数过：
      **`grep -rn TODO src/` 共 23 条，其中只有 8 条是真活**（阶段 2 的 6 条 + 阶段 3 的 2 条），
      另外 15 条是 stale（`run.py` 2 / `utils.py` 3 / `policies.py` 5 / `pg_agent.py` 5）。
      用 TODO 数估剩余工作量会高估近 3 倍。清理留到全部做完，免得和 starter code 的 diff 太乱。

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



**PDF 给的是梯度，代码里却必须写 loss —— 这个转换是我自己造的，不是抄的**

PDF 从式 (4) 一路到式 (22) 给的全是 `∇θ J(θ)`，**通篇没有出现过 actor 的 loss**
（`pdftotext` 后 grep 整份 PDF，唯一一处 "loss" 是 §4.2 要求画 baseline loss 曲线）。
不是它省略了 —— 策略梯度在数学上本来就没有 loss。

loss 是被 PyTorch 的 API 形状倒逼出来的。autodiff 只有一个入口：

```
标量.backward()  →  填好每个 θ.grad  →  optimizer.step() 做 θ ← θ - α·θ.grad
```

它**只会**「对一个标量求导」这一件事，而我手里是一个**已经求好的梯度公式**，
根本不需要它帮我求导。于是只能倒着造一个标量，骗过这个 API。

**反推过程**：要的是梯度上升 `θ ← θ + α∇J`，而 `step()` 做的是减法，
所以需要一个 L 满足 `∇L = -∇J`。代入式 (8)：

```
∇J ≈ (1/N) Σ_i Σ_t  ∇log π(a|s) · A
```

关键在于 **A 与 θ 无关** —— 它是从已采样数据算出的固定数字，
代码里更是 numpy 转回来的，计算图早断了。既然是常数，`∇` 就能提到求和外面：

```
(1/N) Σ ∇log π · A  ==  ∇[ (1/N) Σ log π · A ]
                          └────── 这就是要造的标量 ──────┘
```

取负号即得 `loss = -(log_prob * advantages).mean()`。

这东西叫 **surrogate loss（代理损失）**：**只有梯度有意义，数值本身毫无意义**。
它不是预测误差，不是任何东西的上界，训练成功时也不保证下降。

三个可验证的推论，都能在本仓库里对上：

| 推论 | 证据 |
|---|---|
| `REPORT.md` 全文没要过 actor loss 曲线 | §1.2/§3.2 要的都是 return 曲线；只有 §2.2 要 baseline loss —— 后者是**真**回归损失，该降；前者涨跌都说明不了事 |
| actor 一个 batch 只更新 **1** 次，critic 更新 **5** 次 | `pg_agent.py:82` vs `pg_agent.py:87`（`-bgs` 默认 5） |
| 训练正常时 Actor Loss 经常在**涨** | 别拿它 debug，看 `Eval_AverageReturn` |

第二条的原因值得单独记：`∇L = -∇J` **只在当前 θ、当前这批数据下成立**。
数据是用旧的 π_θ 采的，θ 一动等式就失效，所以走一步就必须重新采样。
critic 那边是真回归损失，同一批数据反复用没问题，所以能走 5 步。
（想在一批数据上多走几步又不出错，就得加重要性采样修正 —— 那就是 PPO。）

**与 hw1 的推导方向正好相反**

| | hw1 行为克隆 | hw2 策略梯度 |
|---|---|---|
| 先有的是 | **loss**（MSE，真实存在的目标） | **梯度**（式 6，log-derivative trick 推出来的） |
| 推导方向 | 目标 → 求导 → 梯度 | 梯度 → 反推 → 一个假目标 |
| loss 数值 | 有意义，该降 | 无意义，别看 |

`hw1/src/hw1_imitation/train.py:160-164` 的「算 loss → zero_grad → backward → step」
四步骨架和这里一模一样，但那边 `compute_loss` 是**真在算损失**，
这里的 loss 是**在拼一个梯度容器**。代码像，语义完全不同。

**边界澄清**（别把话说过头）：`J(θ) = E_{τ~π_θ}[r(τ)]` 是真实存在的目标函数，
只是**在代码里写不出可微表达式** —— θ 藏在采样分布里，而不是被求和的数值里，
这正是式 (4)→(6) 那个 log-derivative trick 要解决的问题。trick 给了梯度，
却没给一个可微的 J。所以准确的说法是：
**J 真实但不可微，L 可微但不真实，两者只在当前 θ 处共享同一个梯度。**

**顺带一个没照抄公式的地方**：式 (8) 除的是 `N`（轨迹条数），
而 `.mean()` 除的是 `batch_size`（时间步总数），两者差一个「平均轨迹长度」的常数因子，
被 learning rate 吸收了。标准做法，但要知道自己偏离了公式：CartPole 里这个因子会随策略变好而变大
（轨迹越来越长），相当于学习率在隐式衰减。

> **这条讲的是「为什么要造一个假 loss」（决策）。**
> 「怎么把式 (8) 一步步翻译成那行代码」（流程 + 形状检查清单）在 `FORMULA_TO_CODE.md`。

**为什么 reward-to-go 方差更低 —— 那一项「期望为 0 但方差不为 0」到底什么意思**

写完两个 estimator 时只知道结论「rtg 方差更小」，没搞懂为什么。补上推导。

把 trajectory-centric 的权重在时刻 $t$ 处劈开成「过去」与「未来」两段：

$$R(\tau)=\underbrace{\sum_{t'<t}\gamma^{t'}r_{t'}}_{P_t\ (\text{过去})}+\underbrace{\sum_{t'\ge t}\gamma^{t'}r_{t'}}_{F_t\ (\text{未来})}$$

于是每个 $(i,t)$ 项拆成两半：

$$\nabla_\theta\log\pi_\theta(a_t|s_t)\cdot R(\tau)=\underbrace{\nabla_\theta\log\pi_\theta(a_t|s_t)\cdot F_t}_{\text{reward-to-go 保留的}}+\underbrace{\nabla_\theta\log\pi_\theta(a_t|s_t)\cdot P_t}_{X_t}$$

**X_t 的期望为零，靠的是 score function 恒等式**：

$P_t$ 只由 $a_t$ **之前**发生的事决定，与 $a_t$ 无关。所以给定历史 $\tau_{<t}$ 时它是个常数，能提到期望外：

$$\mathbb{E}\big[X_t\mid\tau_{<t}\big]=P_t\cdot\mathbb{E}_{a_t\sim\pi_\theta(\cdot\mid s_t)}\big[\nabla_\theta\log\pi_\theta(a_t\mid s_t)\big]$$

而括号里那个恒等于零 —— 就是概率归一化求个导：

$$\mathbb{E}_{a\sim\pi_\theta}[\nabla_\theta\log\pi_\theta(a\mid s)]
=\sum_a \pi_\theta(a\mid s)\frac{\nabla_\theta\pi_\theta(a\mid s)}{\pi_\theta(a\mid s)}
=\sum_a\nabla_\theta\pi_\theta(a\mid s)
=\nabla_\theta\underbrace{\sum_a\pi_\theta(a\mid s)}_{=1}
=\nabla_\theta 1=\mathbf{0}$$

第三个等号是求和与求导交换。再套全期望公式：
$\mathbb{E}[X_t]=\mathbb{E}\big[\mathbb{E}[X_t\mid\tau_{<t}]\big]=\mathbb{E}[P_t\cdot\mathbf{0}]=\mathbf{0}$。

> 这个恒等式是策略梯度的支柱，不止用在这里 —— 阶段 2 的 baseline 能减而不引入偏差
> （PDF 式 12 下面那行 $\nabla_\theta\mathbb{E}[b]=\mathbb{E}[\nabla_\theta\log\pi_\theta\cdot b]=0$）用的是**同一个**恒等式，
> 只是把 $P_t$ 换成了常数 $b$。搞懂一次，两处都通。

**方差不为零 —— 期望是 0 不等于这个量是 0**

$$\operatorname{Var}(X_t)=\mathbb{E}[X_t^2]-\underbrace{(\mathbb{E}[X_t])^2}_{=\,0}=\mathbb{E}\big[\lVert\nabla_\theta\log\pi_\theta(a_t\mid s_t)\rVert^2\,P_t^2\big]>0$$

只消掉了第二项，第一项照样是正的（只要 $P_t\neq 0$）。

具象化：某状态策略 50/50，对某个标量参数 $\nabla_\theta\log\pi=+1$（动作 0）/ $-1$（动作 1），$P_t=10$：

| 采到 | 概率 | $X_t$ |
|---|---|---|
| 动作 0 | 0.5 | $+10$ |
| 动作 1 | 0.5 | $-10$ |

均值 $0$，标准差 $10$。**永远采不到 $X_t=0$，但平均下来是 $0$。** 抛硬币赢 1 输 1 是一回事。
所以它对梯度方向零贡献，只是让你需要更多样本才能把它平均掉。

**在 CartPole 上这个噪声有多大**

$\gamma=1$、每步 $r=1$、$H=200$ 时 $P_t=t$、$F_t=200-t$：

| $t$ | 信号 $F_t$ | 噪声幅度 $\propto P_t$ | 噪信比 |
|---|---|---|---|
| 0 | 200 | 0 | — |
| 100 | 100 | 100 | 1 : 1 |
| 199 | 1 | 199 | **1 : 199** |

**轨迹越长，末尾时刻噪声越压倒信号。而策略变好 ⇒ 轨迹变长 ⇒ 噪声变大。**
这解释了实验里 trajectory-centric 的崩塌为什么总发生在训练**中后期**而不是开头
（`cartpole_lb` 在 20–30 万步塌到 71.33，见 REPORT §1.1）。

**结论**：删掉 $X_t$ 是纯赚 —— $\mathbb{E}[\hat g]$ 一分不变（无偏性保住），$\operatorname{Var}(\hat g)$ 直接砍掉一块，零代价。

> 上面是形式推导。`VARIANCE.md` 里有一个 **4 条轨迹全部列举**的玩具例子，
> 用 `Fraction` 精确算出两个估计量期望都是 $11/2$、方差 $90.75$ vs $30.75$ ——
> 不用统计、不用采样，可以手算复核。那份还整理了本作业四种方差削减技巧
> （rtg / baseline / normalization / GAE）哪些无偏、哪些用偏差换方差。
这就是 PDF §2.2.2 说「we almost always use the second formulation」的原因。

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

**验证「X_t 期望为 0、方差不为 0」（对应上面那条关键决定）**

固定一个未训练的策略，在 CartPole 上采 500 个独立 batch（每 batch ~400 步），
分别算三个梯度估计：`g_traj`（权重 R(τ)）、`g_rtg`（权重 F_t）、`g_past`（权重 P_t）。

判据用**逐坐标 t 统计量** $t=\text{mean}/(\text{std}/\sqrt{N})$，256 个坐标。
H0（该坐标均值为 0）成立时 |t| 应服从 ~N(0,1)，约 5% 的坐标 |t|>2。

| | \|t\|>2 的比例 | max\|t\| | 结论 |
|---|---|---|---|
| 过去项 `g_past` | **0.0%** | **1.25** | 测不出任何非零 → 与「期望为 0」一致 ✅ |
| 信号项 `g_rtg` | 100.0% | 48.72 | 压倒性显著非零（对照组） |

同一批数据、同一个检验，一个 max|t| = 1.25，一个 48.72。

单 batch 梯度的标准差（范数）：

| | 标准差 |
|---|---|
| trajectory-centric | 0.2578 |
| reward-to-go | 0.1102 → **traj 是 rtg 的 2.34 倍** |
| 纯噪声项 X_t 单独 | 0.1784 → 是信号大小（‖E[g_rtg]‖=0.2268）的 **0.8 倍** |

最后一行最说明问题：**X_t 单个 batch 的波动达到信号本身的 80%**，却对梯度方向毫无贡献。

> **踩过的弯路**：一开始用「$\lVert\mathbb{E}[g_\text{past}]\rVert$ 是否随 $N$ 按 $1/\sqrt{N}$ 收缩」来判断，
> 结果 `‖·‖×√N` 在 0.08~0.38 之间乱跳，看不出趋势，差点以为真有偏差。
> **零均值向量的范数是卡方型的，恒为正、不随 N 趋于零** —— 用它判断有没有偏差是错的指标。
> 换成逐坐标 t 检验立刻干净了。以后验证「某个量期望为零」一律用 t 检验，不要用范数。

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

#### 坑 2：把 Adam 的过冲误诊成实现 bug

- **症状**：写完 `MLPPolicyPG.update` 后做方向性验证 —— 喂**单个**数据点，
  给正 advantage，期望 `log π` 上升。四个环境里三个通过，InvertedPendulum 挂了：
  ```
  CartPole     adv=+1: Δlogπ=+0.512 OK    adv=-1: Δlogπ=-1.154 OK
  LunarLander  adv=+1: Δlogπ=+0.736 OK    adv=-1: Δlogπ=-1.112 OK
  InvPendulum  adv=+1: Δlogπ=-0.485 FAIL  adv=-1: Δlogπ=-0.933 OK
  HalfCheetah  adv=+1: Δlogπ=+4.557 OK    adv=-1: Δlogπ=-8.104 OK
  ```
  「给了正的 advantage，这个动作的概率反而降了」看着像符号写反了。
- **排查**：扫学习率，其余一切不变：
  ```
  lr=1e-1   Δlogπ = -9.464   ✗
  lr=1e-2   Δlogπ = -0.485   ✗      ← 默认 5e-3 附近
  lr=1e-3   Δlogπ = +0.018   ✓
  lr=1e-4   Δlogπ = +0.002   ✓
  ```
  lr 一小就正常 ⇒ **不是符号问题，是步子迈过头了**。
- **原因**：Adam 的第一步是 `m̂/(√v̂+ε) ≈ sign(g)`，**每个参数都走满 lr**。
  InvertedPendulum 的 `mean_net` 有 4545 个参数同时各走 0.01，
  合成到 μ 上的位移远超一阶近似的有效范围，直接越过目标点。
  HalfCheetah 的 Δlogπ = +4.56 也印证了步子有多大 —— 只是它恰好没冲过头。
- **教训**：
  1. **单点 + 大 lr 的方向性测试不可靠**。真实训练里 batch 上千、advantage 有正有负，
     不会出现这种单点极端情形。
  2. 想验证「梯度方向对不对」，**别用「跑一步优化器看结果」**，
     直接查解析梯度：`d(loss)/d(log_prob_j)` 应当等于 `-adv_j / batch`。
     这个检验与优化器完全无关，一次就过（见验证记录）。
  3. 遇到「方向反了」先扫 lr 再怀疑符号 —— 两分钟的事，能省掉一轮通读代码。

### 遗留疑问

- [ ] **`.mean()` 除的是 `batch_size` 而不是式 (8) 的 `N`（轨迹条数）**，差一个「平均轨迹长度」
      的因子，被 learning rate 吸收了。标准做法，但 CartPole 里这个因子会随策略变好而变大
      （轨迹从 19.5 步长到 167 步），相当于**学习率在隐式衰减**。
      这是好事还是坏事？跟 `-na` 的作用有没有重叠？没想清楚。
- [ ] **梯度方差在训练中涨了约 6000 倍**（`VARIANCE.md` 实测：0.055 → 339.9），
      而学习率始终不变。这解释了崩塌为什么发生在中后期。
      那么除了 `-na`，直接给学习率加个衰减是不是也能缓解？没试。
- [x] ~~为什么 rtg 方差更低~~ —— 已补齐，见上面「关键决定」与 `VARIANCE.md`。

---

## 阶段 2 — Neural Network Baseline（PDF §4）

跑通目标：HalfCheetah `cheetah_baseline` 末尾 Eval return > 300。
**实测 316.90，一次通过。**

### TODO 清单

- [x] `critics.py` `ValueCritic.forward` ✅
- [x] `critics.py` `ValueCritic.update`：loss + optimizer step ✅
- [x] `pg_agent.py` `_estimate_advantage`：跑 critic 拿 values ✅
- [x] `pg_agent.py` `_estimate_advantage`：有 baseline、无 GAE 时的 advantages ✅
- [x] `pg_agent.py` `update` step 4：做 `baseline_gradient_steps` 次 critic 更新 ✅

### 关键决定

**critic 的训练目标是蒙特卡洛回报，不是 bootstrap**

`update(obs, q_values)` 里的 `q_values` 来自 `_calculate_q_vals`，
docstring 写得很直白：**"Monte Carlo estimation of the Q function"** ——
**从实际收到的奖励折扣加总，没有任何网络参与**。

所以 critic 是个**纯监督回归器**：输入状态、标签是「从该状态实际拿到的回报」、
loss 是 MSE。和 hw1 的行为克隆结构一样，只是标签从「专家动作」换成「实际回报」。

| | Monte Carlo（本阶段） | Bootstrap / TD |
|---|---|---|
| 目标 | $\sum_{t'\ge t}\gamma^{t'-t}r_{t'}$ | $r_t+\gamma V_\phi(s_{t+1})$ |
| 用到自己吗 | ❌ 纯真实奖励 | ✅ 用自己对下一状态的预测 |
| 偏差 / 方差 | 无偏 / 大 | 有偏 / 小 |

**但阶段 3 会引入 bootstrap** —— GAE 的 $\delta_t=r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t)$
里那半截就是 TD target。这正是 `VARIANCE.md` 四技巧表里 GAE 那行「有偏」的来源，
$\lambda$ 就是控制「用多少 bootstrap」的旋钮。

> **一个只有读代码才发现的细节**：本作业里 critic 的训练目标**始终**是 MC，
> 即使到了阶段 3 —— step 4 传的是 `q_values` 而不是 GAE 的 λ-return。
> 真实的 PPO 实现通常把 critic 也回归到 `returns = advantages + values`，
> 那个目标本身就是 bootstrap 的。两种都能用，作业选了简单的。

**目标的合法性依赖 `-rtg`**

PDF 式 (13) 里 $V^\pi(s_t)$ 的求和是**从 $t$ 开始**的，所以只有
`_discounted_reward_to_go` 那一版才是合法目标。`_discounted_return` 那版
（整条轨迹回报，每个 $t$ 相同）含了 $t$ 之前的奖励，不是 $V^\pi(s_t)$。

**代码不强制这一点** —— `--use_baseline` 不带 `-rtg` 也能跑，critic 会安静地
回归到错误目标。作业里 exp2 的命令带了 `-rtg`，所以没事。这个坑写进
`critics.py` `update` 的 docstring 了。

**`advantages = q_values - values` 必须全程 numpy**

`_estimate_advantage` 的 docstring 明写 "Operates on flat 1D NumPy arrays"，
但中间要问 critic（torch）。所以是 **numpy 进、借道 torch、numpy 出**：

```python
values = ptu.to_numpy(self.critic(ptu.from_numpy(obs)))
```

这不只是类型体操，是**在结构上保证 advantage 不带计算图**。
`VARIANCE.md` 整套推导的前提是 $A$ 与 $\theta$ 无关，$\nabla$ 才能提到求和外：

$$\frac1N\sum\nabla\log\pi\cdot A=\nabla\Big[\frac1N\sum\log\pi\cdot A\Big]$$

若 $A$ 带计算图，$\nabla(\log\pi\cdot A)$ 会多出 $\log\pi\cdot\nabla A$ 一项，
推导失效；而且 actor 的 `loss.backward()` 会把梯度灌进 critic ——
critic 本该只学回归，却被推去最大化 return。
**实测正确写法下，actor backward 之后 critic 梯度为 `None`，两网络完全隔离。**

**actor 走 1 步、critic 走 `-bgs` 步的不对称**（阶段 1 已推导，这里落到代码）

actor 的 $\nabla L=-\nabla J$ 只在当前 $\theta$、当前这批数据下成立，走一步就得重采样；
critic 的 loss 是真回归损失，`(obs, q_values)` 就是个固定数据集，反复用完全合法。
而且**必须多走几次** —— 实测 `-bgs` 5→1，critic 预测误差差 **8 倍**。

### 验证记录

`ValueCritic` 单元验证（回归一个已知函数 $V(s)=10s_0$，200 步）：

| 检查 | 结果 |
|---|---|
| 真的学会了吗 | loss $105.7\to0.032$，平均绝对误差 **0.085**（真值范围 ±25）✅ |
| loss 值对吗（对照手算 MSE） | `0.486065` vs `0.486065` ✅ |
| assert 挡得住吗 | 喂 `(7,1)` 标签 → `AssertionError: (Size([7]), Size([7,1]))` ✅ |
| `zero_grad` 生效吗 | 1 次 `0.165` / 2 次 `0.138`，不是 2× ✅ |
| 标签是常数吗 | `requires_grad=False` ✅ |

**不 squeeze 的代价**（同一回归任务，唯一区别是 loss 里 squeeze 与否）：

| | 最终 loss | 预测与真值平均绝对误差 |
|---|---|---|
| squeeze 了 `(B,)` | 0.032 | **0.085** |
| 没 squeeze `(B,1)` | 104.2 | **8.307** |

没 squeeze 那版**根本没学会** —— MSE 广播成 $(B,B)$，它在最小化「每个预测 vs 每个目标」
的两两误差，最优解是让预测恒等于目标的均值，网络退化成输出常数。

`-bgs` 的效果（同一目标，100 轮，每轮换新数据）：

| `-bgs` | 第 1 轮报出 | 第 100 轮报出 | 预测误差 |
|---|---|---|---|
| 1 | 30.74 | 0.353 | 0.456 |
| 5 | 16.70 | 0.0028 | 0.056 |
| 20 | 2.72 | 0.0001 | 0.014 |

端到端：`use_baseline=True` 时 `pg_agent.update` 返回
`{'Actor Loss': -2.5869, 'Baseline Loss': 0.8746}`，两条 loss 都进 `log.csv`。

### 踩的坑

#### 预告的两条，实际都没撞上 —— 而且第二条的预告是错的

阶段 1 结束时预埋了两条「已知会拦你的」，实测：

| 预告 | 实际 |
|---|---|
| `assert values.shape == q_values.shape` 会撞（网络输出 `(batch,1)`） | **没撞** —— `forward` 里提前 squeeze 了，assert 直接过 |
| values 忘转 numpy 导致梯度串到 actor，**「这个不报错」** | ❌ **预告错了**。实测 `q_values - values_tensor` 抛 `TypeError: unsupported operand type(s) for -: 'numpy.ndarray' and 'Tensor'`，**会崩** |

第二条值得记：**numpy 在左、tensor 在右时，`numpy.__sub__` 不认识 Tensor，直接报错。**
如果顺序反过来（`tensor - numpy`）才会静默产出带计算图的 tensor。
本作业的写法是 `q_values - values`，正好是安全的那个方向。

> **教训**：预埋的坑也要标「这是预测还是实测」。这条预告写得像实测结论，
> 差点让我以为躲过了一个静默 bug，其实那个 bug 在这个方向上根本不存在。

#### 坑：`self.critic(obs)` 直接喂 numpy

第一次写成 `values = self.critic(obs)`，`obs` 是 numpy：

```
TypeError: linear(): argument 'input' (position 1) must be Tensor, not numpy.ndarray
```

会崩，安全。修法是两次转换（见「关键决定」）。

### 遗留疑问

- [ ] **`-blr 0.001` 末期 loss 已追上默认配置**（19.12 vs 19.07），但最终 return
      差了 460（-146 vs +317）。我在 REPORT §2.3 给的解释是「value target 非平稳，
      critic 早期跟踪慢把 actor 带进坏轨迹，后来追上也回不来」。
      **这个解释合理但没验证** —— 要验的话得看早期（前 20 轮）的 advantage 质量，
      比如 corr(advantage, 真实 advantage)。没做。
- [ ] critic 的**训练误差 vs 泛化误差**没区分。step 4 循环里取的是最后一次 loss，
      那是拟合完这批数据后的**训练误差**。取第一次（没见过这批数据时）更像泛化误差。
      两条曲线会不会讲不同的故事？没试。
- [ ] `VARIANCE.md` 记了「$V^\pi$ 不是方差最小的 baseline，最优的是
      $\lVert\nabla\log\pi\rVert^2$ 加权的回报」。那个最优 baseline 在这里能不能算、
      算了有多少提升？没试。
---

## 阶段 3 — Generalized Advantage Estimation（PDF §5）

跑通目标：LunarLander 五个 λ 里最好的一个，训练中至少出现一次 Eval return > 150。

### TODO 清单

- [x] `pg_agent.py` `_estimate_advantage`：GAE 倒序递推 ✅ 四组钉桩全过

### 关键决定

**式 (18)→(22) 的五步变形：每一步在干什么**

$$\underbrace{(18)}_{\substack{\text{有限 }H\\ \text{带归一化}}}
\xrightarrow[\ \lambda^{H-t-1}\to0\ ]{H\to\infty}
\underbrace{(19)}_{\text{只剩 }(1-\lambda)}
\xrightarrow[\ A_n=\sum\gamma^k\delta\ ]{\text{交换求和次序}}
\underbrace{(20)}_{(\gamma\lambda)^k\delta}
\xrightarrow[\ \text{截断到 }H\ ]{}
\underbrace{(21)}_{\text{有限和}}
\xrightarrow{\text{提公因子}}
\underbrace{(22)}_{\text{递推},\ O(H)}$$

**要实现的是最右边那个。** 中间四步是等价变形，PDF 列出来是为了说明「凭什么可以这么算」。

**(18)→(19)：归一化常数塌缩**

$\lambda<1$，所以 $H\to\infty$ 时 $\lambda^{H-t-1}\to 0$：

$$\frac{1-\lambda}{1-\lambda^{H-t-1}}\;\longrightarrow\;\frac{1-\lambda}{1}=1-\lambda$$

式 (19) 前面那个孤零零的 $(1-\lambda)$ 就是这么来的。

**(19)→(20)：γ 去哪了？—— 它一直藏在 $A_n$ 内部**

式 (19) 里只有 $\lambda^{n-1}$ 的权重衰减，看不到 $\gamma$；式 (20) 却冒出 $(\gamma\lambda)^k$。
原因是这个恒等式 —— **n-step advantage 本身就是 $\delta$ 的 $\gamma$-折扣和**：

$$A_n(s_t)=\sum_{k=0}^{n-1}\gamma^k\delta_{t+k}$$

靠 telescoping 证：

$$\sum_{k=0}^{n-1}\gamma^k\delta_{t+k}=\sum_k\gamma^k\big[r_{t+k}+\gamma V(s_{t+k+1})-V(s_{t+k})\big]$$
$$=\sum_k\gamma^k r_{t+k}+\underbrace{\sum_{j=1}^{n}\gamma^{j}V(s_{t+j})-\sum_{j=0}^{n-1}\gamma^{j}V(s_{t+j})}_{\text{中间项两两抵消，只剩首尾}}
=\sum_k\gamma^k r_{t+k}+\gamma^n V(s_{t+n})-V(s_t)$$

右边正是 $A_n$ 的定义（式 17）。**中间所有 $V$ 都消掉了。**

代进式 (19) 并交换求和次序（固定 $k$，有贡献的是 $n\ge k+1$）：

$$A^{\text{GAE}}=(1-\lambda)\sum_{k=0}^{\infty}\gamma^k\delta_{t+k}\underbrace{\sum_{n=k+1}^{\infty}\lambda^{n-1}}_{=\ \lambda^k/(1-\lambda)}=\sum_{k=0}^{\infty}(\gamma\lambda)^k\delta_{t+k}$$

$(1-\lambda)$ 被几何级数求和抵消，$\gamma^k\lambda^k$ 合并成 $(\gamma\lambda)^k$。

**$\gamma$ 和 $\lambda$ 的分工**（到式 20 才合并，不代表它们本来是一回事）：

| | 作用 | 在哪层 |
|---|---|---|
| $\gamma$ | **时间折扣** —— 未来的奖励值多少钱 | 每个 $A_n$ **内部** |
| $\lambda$ | **估计量加权** —— 用多少 1 步、多少 2 步…… | $A_n$ **之间** |

**数值验证**（$\gamma{=}0.9,\lambda{=}0.95$，$r=[1,2,0,3,1,2]$，$V=[5,4,6,2,3,1]$）：

| 检查 | 结果 |
|---|---|
| $A_n(s_t)\overset{?}{=}\sum\gamma^k\delta_{t+k}$，全部 $(t,n)$ 组合 | 全部一致 |
| $(1-\lambda)\sum_n\lambda^{n-1}A_n\overset{?}{=}\sum_k(\gamma\lambda)^k\delta_{t+k}$ | 差 $2\times10^{-15}$（浮点精度） |

**不要按式 (18) 字面实现 —— 那是 $O(H^3)$**

式 (18) 的直接翻译是「先算出所有 n-step advantage $A_n$，再加权平均」：

$$A^{\text{GAE}}(s_t)=\frac{1-\lambda}{1-\lambda^{H-t-1}}\sum_{n=1}^{H-t-1}\lambda^{n-1}A_n(s_t)$$

对每个 $t$ 要算 $H-t$ 个 $A_n$，每个 $A_n$ 本身是 $n$ 项求和 → **$O(H^3)$**。
LunarLander 的 `--ep_len 1000` 下是 $10^9$ 次运算，一条轨迹就跑不动。

式 (20)→(21) 的恒等式把整个加权平均塌成对 $\delta$ 的几何加权和：

$$A^{\text{GAE}}(s_t)=\sum_{t'=t}^{H-1}(\gamma\lambda)^{t'-t}\delta_{t'}
\quad\Longrightarrow\quad A_t=\delta_t+\gamma\lambda A_{t+1}$$

倒扫一遍，**$O(H)$**。数值验证过（$\gamma{=}0.9,\lambda{=}0.95$，$r=[1,2,0,3]$，$V=[5,4,6,2]$）：

| | 结果 |
|---|---|
| 加权所有 $A_n$（式 18） | `[0.0617, 0.54, -3.345, 1.0]` |
| 倒序递推（式 22） | `[0.0617, 0.54, -3.345, 1.0]` |

完全一致。**递推法根本不生成 $A_n$ 那张中间表。**

> 这和阶段 1 `_discounted_reward_to_go` 是**同一个模式** —— 正序 $O(H^2)$ vs 倒序 $O(H)$，
> 区别只是这次被折叠的是 $\delta$ 而不是 $r$。
> 「遇到后缀求和就找递推」在这份作业里出现了两次。

**单步循环体只需要相邻两项**

```
δ_i = r_i + γ·V(s_{i+1}) − V(s_i)      ← 一步 TD 误差
A_i = δ_i + γλ·A_{i+1}                  ← 用上一轮刚算出的结果
```

用到的全是**当前 i 和 i+1**，没有任何跨多步的求和。这就是为什么循环外要预留两格：

```python
values = np.append(values, [0])           # 让 V(s_{i+1}) 在 i 是最后一行时不越界
advantages = np.zeros(batch_size + 1)     # 让 A_{i+1} 同理取到 0
```

**又是阶段 1 `running = 0.0` 那一招** —— 用虚拟边界值免掉特判。三次了。

**`batch_size` 是拍平后的时间步总数，不是轨迹条数**

```python
batch_size = obs.shape[0]
```

| | 值 |
|---|---|
| `-b` / `args.batch_size` | **最少**要采多少步 |
| `obs.shape[0]` | 实际总步数，**≥ `-b`**（轨迹不截断，跑完才停） |

实测：HalfCheetah `-b 5000` 每轮恰好 5000（轨迹固定 1000 步，整除）；
CartPole `-b 1000` 是 `1012, 1010, 1016, 1038, 1099…`，范围 `[1000, 1199]` ——
**轨迹长度可变，每轮 batch_size 都不同**。

所以循环扫的是**一条跨越多条轨迹的长数组**：

```
obs       [轨迹1 的 T₁ 步 | 轨迹2 的 T₂ 步 | 轨迹3 的 T₃ 步]
terminals [0,0,…,0,1     | 0,0,…,0,1     | 0,0,…,0,1     ]
                      ↑ 边界          ↑ 边界
```

`for i in reversed(range(batch_size))` 从最后一条轨迹的末尾一路倒扫到第一条的开头。
**不管边界的话，轨迹 2 的第一步会去用轨迹 3 的 advantage。**
这就是 `terminals` 存在的全部理由 —— 阶段 1 记的那句
「语义边界被搬进了 `terminals`，阶段 3 的 GAE 递推就必须靠它断开」在这里兑现。

**`terminals[i]==1` 时两处都要归零**

该行是所在轨迹的最后一步，`values[i+1]` 和 `advantages[i+1]` 都属于**下一条轨迹**，不能用。
PDF 式 (16) 下面给的边界情形 $\delta_{H-1}=r_{H-1}-V_\phi(s_{H-1})$ 就是把 $\gamma V_\phi(s_{t+1})$ 那项去掉。

两种写法：乘掩码 `* (1 - terminals[i])`，或显式 `if`。
**掩码更简洁，且是生产代码的普遍写法**（verl / torchrl 的 GAE 都是掩码版）。

### 验证记录

四组钉桩，全部通过（$\gamma=0.9$，用 Stub 替换 critic 以便手算对答案）：

| 用例 | 期望 | 结果 |
|---|---|---|
| 单轨迹 $r{=}[1,2,0,3]$, $V{=}[5,4,6,2]$, $\lambda{=}0.95$ | `[0.0617, 0.54, -3.345, 1.0]` | ✅ |
| **双轨迹** `terminals=[0,0,1,0,1]` | `[0.3139, 0.835, -3.0, 6.59, -2.0]` | ✅ |
| $\lambda=0$ | 逐元素等于单步 TD 残差 $\delta$ | ✅ |
| $\lambda=1$ | 逐元素等于纯 MC 的 $Q-V$ | ✅ |

**第二组是唯一能验出边界 bug 的** —— 单轨迹用例里 `terminals` 只有末位是 1，
掩码写不写都得到同样结果。这是阶段 1「坑 1」那条教训的复用：
**验证用例必须能区分正确实现和错误实现。**

后两组同时验证了代码和对 PDF §5 Q2 的理解，直接引进了 `REPORT.md` §3.3 Q2 当证据。

### 踩的坑

#### 坑：递推里把「写进格子」写成了「换掉整个数组」

- **症状**：`IndexError: invalid index to scalar variable`，两组用例都崩。
- **原因**：漏了一个下标。
  ```python
  advantages    = delta + γλ * advantages[i+1] * mask   # ✗ 覆盖整个数组
  advantages[i] = delta + γλ * advantages[i+1] * mask   # ✓ 写进第 i 格
  ```
  第 1 轮 `advantages` 从 `(B+1,)` 的数组变成一个 float；
  第 2 轮 `advantages[i+1]` 去索引标量 → 崩。
- **根因**：**递推的本质是「同一个数组既读又写」** ——
  `advantages[i] = …` 写当前格，`advantages[i+1]` 读上一轮写的格。
  写成 `advantages = …` 就把「填格子」变成了「换数组」，递推链当场断掉。
- **为什么阶段 1 没遇到**：`_discounted_reward_to_go` 用的是 `running` 这个**单独的标量**
  做累积器，不需要原地改数组。这里因为下游要全部 $A_i$，必须写数组 —— 是新的一类错误。
- **教训**：
  1. 好消息是**它会崩**，不是静默。
  2. 但 **`batch_size == 1` 时循环只跑一轮，永远撞不上** ——
     又一个「验证用例长度必须 > 1」的例子。

### 遗留疑问

- [x] ~~式 (18) 的归一化常数 $\frac{1-\lambda}{1-\lambda^{H-t-1}}$ 在式 (21) 的递推实现里去哪了？~~
      **答**：PDF 从 (19) 到 (21) **换了 horizon 假设，没有严格保持归一化**。
      (18) 是有限 $H$、权重严格和为 1；(19)(20) 取 $H\to\infty$，常数塌成 $1-\lambda$；
      (21) 再把 (20) 的无穷和**截断回 $H-1$**。
      所以**式 (21) 不完全等于式 (18)** —— 它丢掉了 $t'\ge H$ 的尾巴。
      实践中无所谓：$(\gamma\lambda)^{H-t}$ 在轨迹够长时已极小，且末尾用 $V(s_H)=0$ 收尾。
      **所有主流实现（verl / torchrl / SB3）都用 (21)/(22)，没人管那个归一化常数。**

- [ ] **五个 λ 的 `Baseline Loss` 差 47 倍**（λ=0 是 3936、λ=0.95 是 83），但 critic 的
      回归目标 `q_values` **与 λ 无关**。我在 REPORT §3.1 的解释是「λ 改变策略 →
      策略改变轨迹 → 回报分布不同 → MSE 不在一个量纲上」。**合理但没验证** ——
      要验的话得比较五个 run 的 `q_values` 分布（均值、方差），没做。

- [ ] **λ=0.98 和 λ=1 都「冲上去又掉下来」，λ=0.99 却稳住了。** 单 seed，
      这个区别有多少是真信号、有多少是运气？PDF 自己也说 "Results may have some variance"。
      跑多 seed 才说得清，作业只要求一次。
---

## 阶段 4 — 调参（PDF §6）

无代码改动。调参**结果**记在 `REPORT.md` §4，**方法论**记在 `ABLATION.md`，这里只记进度和待办。

### 🔄 当前进度：做了一半

**已完成**：

- [x] 默认基线 `pendulum`（`-n 100 -b 5000`，500K 步）—— §4.4 的对照图要用
- [x] L9(3⁴) 正交表九组（`run_experiments.sh exp4-l9`），4 因子 × 3 水平：
      discount / 网络 `-l/-s` / batch / lr。固定 `-rtg -na --use_baseline --gae_lambda 0.99`
- [x] 主效应分析（`analyze_l9.py`）、九组 3×3 曲线（`plot_results.py exp4-l9`）
- [x] 方法论写进 `ABLATION.md`

**没做完 —— 所以现在还不能说 exp4 是「系统性完成」的**：

- [ ] **确认实验**。主效应预测的最优组合是
      $(\gamma{=}0.99,\ \text{网络}\ 2/32,\ b{=}1000,\ lr{=}1\text{e-}2)$，
      **它不在跑过的九组里**（最接近的是 #4，只差 lr）。
      正交表给的是**预测**，没验证就不能当结论 —— 见 `ABLATION.md` 第 7 步。
- [ ] **交互效应没查**。`lr=2e-2` 的组内极差 **80%**（#4 优秀 80%、#8 崩溃 8%、#3 未达标 0%），
      是强交互的信号。L9 只能估主效应，自由度已用光。
      要看清得走两阶段：batch × lr 做 3×3 全因子（其他因子固定在主效应最优档）。
- [ ] **因子 7（要不要 GAE）没消融**。九组全都开着 `--use_baseline --gae_lambda 0.99`，
      所以这个设计**回答不了「GAE 在 InvertedPendulum 上到底有没有用」**。
      需要在最优配置上做三档对照：无 baseline / baseline 无 GAE / baseline+GAE。
- [ ] `REPORT.md` §4.1 表格、§4.2 最佳配置、§4.3 问答、§4.4 图 —— 全空。

**结论先别下。** 目前九组里综合最好的是 #4（`γ=0.99, 2/32, b1000, lr2e-2`，
31,405 步达标、末 10 均 1000、达标后 80% 时间 ≥950），但那是**九组中的最好**，
不是**主效应预测的最优**，两者未必一致。

### 单 seed 的局限

全部九组只跑了 `seed=1`。PDF §6 的 deliverable 自己说
"results may vary with different random seeds"，且只要求提交最好的一次。
但这意味着**主效应表里每个数都只由 3 个单次运行支撑**，
组内极差里有多少是真交互、多少是 seed 噪声，分不清。

---

## 跨作业复用

> 这节留到 hw2 全部做完再回填 —— 挑出对 hw3（DQN / SAC）、hw5 有用的结论。

> TODO
