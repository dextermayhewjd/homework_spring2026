# 方差：策略梯度的核心问题

> **这份记什么**：一个小到能把所有情况列全的例子，把「删掉一个期望为零的项 ——
> 期望不变、方差变小」这件事算到底。以及为什么方差在 RL 里比在监督学习里危险得多。
>
> **不记什么**：
> - 「公式怎么翻译成代码」→ `FORMULA_TO_CODE.md`
> - 「张量形状怎么想」→ `SHAPES.md`
> - 「hw2 这次为什么这么决定」→ `NOTES.md`（阶段 1 有 score function 恒等式的形式推导）
> - 「跑出来多少」→ `REPORT.md`
>
> 和另外两份 reference 一样，这是**跨作业复用**的：hw2 的四种技巧、hw3 的 target network、
> final project 的 PPO/GRPO，全都在解同一个问题。

---

## 核心命题

先把三个量定义清楚。对**一条**轨迹 $\tau$ 的时刻 $t$：

| 记号 | 定义 | 含义 |
|---|---|---|
| $P_t$ | $\sum_{t'<t}\gamma^{t'}r_{t'}$ | $t$ **之前**已经到手的奖励（$P_0=0$） |
| $F_t$ | $\sum_{t'\ge t}\gamma^{t'}r_{t'}$ | $t$ **及之后**的奖励，即 reward-to-go |
| $R(\tau)$ | $P_t+F_t$ | 整条轨迹的回报，对每个 $t$ 都是同一个数 |

$$\boxed{X_t:=\nabla_\theta\log\pi_\theta(a_t\mid s_t)\cdot P_t}$$

$X_t$ 是**单个时间步**上被多算的那一项，是个随机向量（与 $\theta$ 同维），
随机性来自「采样到哪条轨迹」。两个估计量的关系：

$$\hat g_{\text{traj}}=\sum_t\nabla_\theta\log\pi_\theta(a_t|s_t)\,R(\tau)
=\underbrace{\sum_t\nabla_\theta\log\pi_\theta(a_t|s_t)F_t}_{\hat g_{\text{rtg}}}+\sum_t X_t$$

$$\boxed{\hat g_{\text{traj}}=\hat g_{\text{rtg}}+\sum_t X_t,\qquad \mathbb{E}[X_t]=0\ \ \forall t,\qquad \operatorname{Var}\Big(\sum_t X_t\Big)>0}$$

**注意是对 $t$ 求和**，不是单独一项。期望是线性的，所以
$\mathbb{E}[\sum_t X_t]=\sum_t\mathbb{E}[X_t]=0$；方差不是线性的，加不掉。
（不同 $t$ 的 $X_t$ 来自同一条轨迹，并不独立，所以方差也**不能**简单相加。）

<details>
<summary><b>不放心的话，把 $T=3$ 完全展开验一遍（点开）</b></summary>

记 $s_t:=\nabla_\theta\log\pi_\theta(a_t|s_t)$，$\gamma=1$，$R=r_0+r_1+r_2$。

**左边**（每一项都乘同一个 $R$）：$\hat g_{\text{traj}}=s_0R+s_1R+s_2R$

**右边**，先列出两半：

| $t$ | $F_t$（未来） | $P_t$（过去） |
|---|---|---|
| 0 | $r_0+r_1+r_2$ | $0$ |
| 1 | $r_1+r_2$ | $r_0$ |
| 2 | $r_2$ | $r_0+r_1$ |

$$\hat g_{\text{rtg}}=s_0(r_0{+}r_1{+}r_2)+s_1(r_1{+}r_2)+s_2(r_2)$$
$$\textstyle\sum_t X_t=s_0\cdot 0+s_1\cdot r_0+s_2\cdot(r_0{+}r_1)$$

**逐行相加，每一行都拼回完整的 $R$**：

$$\begin{aligned}
s_0:&\quad (r_0{+}r_1{+}r_2)+0 &&= R\\
s_1:&\quad (r_1{+}r_2)+r_0 &&= R\\
s_2:&\quad (r_2)+(r_0{+}r_1) &&= R
\end{aligned}$$

所以右边 $=s_0R+s_1R+s_2R=$ 左边。**这是恒等变形，没有任何近似。**

真实轨迹上也验过：CartPole $T=30$，直接算 $\hat g_{\text{traj}}$ 与算 $\hat g_{\text{rtg}}+\sum_t X_t$，
全部 4610 个梯度分量逐元素比对，最大差 $6.2\times 10^{-6}$（浮点误差量级），`torch.allclose` 通过。

**最容易卡住的点**：$R(\tau)$ 是一个固定的数，$P_t$ 和 $F_t$ 却随 $t$ 变 ——
怎么可能对每个 $t$ 都有 $P_t+F_t=R(\tau)$？因为两者**此消彼长**：
$t$ 往后走一步，$P_t$ 多吃一个奖励、$F_t$ 就少一个，和不变。

</details>

> **$X_0$ 恒为 0**：第 0 步之前没有任何奖励，$P_0=0$。
> 所以下面 $H=2$ 的玩具例子里求和塌成单独一项 $X_1$ —— 表格里没有 $X_0$ 列不是省略了。

### 真实轨迹上 $X_t$ 长什么样

一条 CartPole 轨迹（$T=35$，$\gamma=1$，每步 $r=1$，所以 $P_t=t$、$F_t=35-t$）：

| $t$ | $P_t$ | $F_t$ | $\lVert X_t\rVert$ | 噪信比 $P_t/F_t$ |
|---|---|---|---|---|
| 0 | 0 | 35 | **0.0000** | 0.00 |
| 1 | 1 | 34 | 0.0736 | 0.03 |
| 3 | 3 | 32 | 0.2392 | 0.09 |
| 33 | 33 | 2 | 6.2491 | 16.50 |
| 34 | 34 | 1 | **6.2634** | **34.00** |

开头这一项几乎不存在，到末尾变成信号的 34 倍。**轨迹越长，末段被污染得越厉害。**

> **措辞小心**：整条轨迹上 $\sum_t\lVert X_t\rVert=59.47$、$\sum_t\lVert\nabla\log\pi\cdot F_t\rVert=39.16$，
> 但这是**每项范数之和**，不是**向量和的范数**。$X_t$ 求和时会大量互相抵消（期望为零），
> 所以不能说「最终梯度里 60% 是噪声」。
> 正确的理解是：**每一项的大小才是方差的来源** —— 抵消得再好，单次采样的抖动也由项的大小决定。

所以删掉 $\sum_t X_t$：

- **期望一分不变** —— 你估计的还是同一个东西，无偏性保住
- **方差变小** —— 估计更准
- **代价为零**

`NOTES.md` 阶段 1 有 score function 恒等式 $\mathbb{E}_{a\sim\pi}[\nabla_\theta\log\pi_\theta(a|s)]=0$
的形式推导。这里不重复，改用一个能手算验证的例子。

---

## 玩具例子：4 条轨迹，全部列举

**环境**（2 步，每步 2 个动作，$\gamma=1$）：

| | 选动作 0 | 选动作 1 |
|---|---|---|
| 第 0 步奖励 $r_0$ | **10** | 0 |
| 第 1 步奖励 $r_1$ | **1** | 0 |

**策略**：每步 50/50。记 score $s(a)=\nabla_\theta\log\pi_\theta(a)$，下面取 $s(0)=+1$、$s(1)=-1$。

<details>
<summary><b>这两个数从哪来的？（点开）</b></summary>

代码里离散策略走的是 `Categorical(logits=z)`，即 softmax：

$$\pi(a)=\frac{e^{z_a}}{\sum_j e^{z_j}}$$

**第一步：先取 log，把除法变减法。** 这是全部的关键 —— 原式是分式，直接求导要用商法则；
取 log 后变成「一个简单项减一个 log-sum-exp」，两项可以分开处理：

$$\log\pi(a)=\log e^{z_a}-\log\sum_j e^{z_j}=z_a-\log\sum_j e^{z_j}$$

**第二步：对 $z_k$ 求导，两项分别算。**

第一项 —— $z_a$ 是向量的第 $a$ 个分量，对第 $k$ 个分量求导：

$$\frac{\partial z_a}{\partial z_k}=\begin{cases}1 & k=a\\ 0 & k\ne a\end{cases}=\mathbb{1}[k=a]$$

第二项 —— 记 $S=\sum_j e^{z_j}$。这一步拆细了写（以 3 个动作为例，$z=(z_0,z_1,z_2)$）。

**前提**：$z_0,z_1,z_2$ 是网络输出向量的三个**互不相干的数**，所以

$$\frac{\partial z_j}{\partial z_k}=\begin{cases}1 & j=k\quad\text{（对自己求导）}\\[2pt] 0 & j\ne k\quad\text{（对别人求导，别人是常数）}\end{cases}$$

后面每一步都建立在这条上。

**(1) $S$ 是什么** —— 就是三项相加：

$$S=\sum_j e^{z_j}=e^{z_0}+e^{z_1}+e^{z_2}$$

**(2) 对 $z_k$ 求导，逐项处理。** 求导是线性的，和的导数 = 导数的和：

$$\frac{\partial S}{\partial z_k}=\frac{\partial e^{z_0}}{\partial z_k}+\frac{\partial e^{z_1}}{\partial z_k}+\frac{\partial e^{z_2}}{\partial z_k}$$

单看一项，链式法则（外层 $e^u$、内层 $u=z_j$）：

$$\frac{\partial e^{z_j}}{\partial z_k}=e^{z_j}\cdot\underbrace{\frac{\partial z_j}{\partial z_k}}_{\text{前提那条}}=\begin{cases}e^{z_j} & j=k\\[2pt] 0 & j\ne k\end{cases}$$

具体代 $k=1$ 看：

$$\frac{\partial S}{\partial z_1}=\underbrace{e^{z_0}\cdot 0}_{e^{z_0}\text{ 里没有 }z_1}+\underbrace{e^{z_1}\cdot 1}_{\text{只有这项含 }z_1}+\underbrace{e^{z_2}\cdot 0}_{e^{z_2}\text{ 里没有 }z_1}=e^{z_1}$$

**三项里只活下来下标匹配的那一项**，所以一般地 $\dfrac{\partial S}{\partial z_k}=e^{z_k}$。

**(3) log 的链式法则**：$\dfrac{d}{dx}\log u=\dfrac{1}{u}\cdot\dfrac{du}{dx}$，此处 $u=S$、$x=z_k$：

$$\frac{\partial\log S}{\partial z_k}=\frac{1}{S}\cdot\frac{\partial S}{\partial z_k}$$

**(4) 代入 (2) 的结果**：

$$\frac{\partial\log S}{\partial z_k}=\frac{1}{S}\cdot e^{z_k}=\frac{e^{z_k}}{S}$$

**(5) 认出这是什么。** 把 $S$ 展开回去：

$$\frac{e^{z_k}}{S}=\frac{e^{z_k}}{\sum_j e^{z_j}}\;\overset{\text{softmax 的定义}}{=}\;\pi(k)$$

不是「凑巧等于」，是**同一个表达式**。

$$\boxed{\frac{\partial\log S}{\partial z_k}=\pi(k)}$$

**代数字验一遍**（$z=(1,0,2)$，对 $z_1$ 求导）：

| 步骤 | 计算 | 结果 |
|---|---|---|
| (1) | $S=e^1+e^0+e^2=2.71828+1+7.38906$ | $11.10734$ |
| (2) | $\partial S/\partial z_1=e^1\!\cdot\!0+e^0\!\cdot\!1+e^2\!\cdot\!0$ | $1.00000$ |
| (3)(4) | $(1/11.10734)\times 1.00000$ | $0.090031$ |
| (5) | $\pi(1)=e^{z_1}/S=1/11.10734$ | $0.090031$ ✓ |

autograd 也对得上：`d(logsumexp)/dz = [0.244728, 0.090031, 0.665241]`
与 `softmax(z) = [0.244728, 0.090031, 0.665241]` 逐元素相等。

> **log-sum-exp 的导数恰好就是 softmax。** 值得单独记住，写 attention、
> 算 log-partition 的梯度时还会遇到。

**第三步：合并。**

$$\boxed{\frac{\partial\log\pi(a)}{\partial z_k}=\mathbb{1}[k=a]-\pi(k)}$$

**这个式子的含义**：$\underbrace{\mathbb{1}[k=a]}_{\text{实际选了 }k\text{ 吗}}-\underbrace{\pi(k)}_{\text{本来预期选 }k\text{ 的概率}}$
= **实际 − 预期 = 惊讶程度**。

- **真选了 $k$**：抬高 $z_k$ 让它更可能，梯度 $=1-\pi(k)>0$。本来就几乎必选时梯度趋于 0（抬也没用）
- **没选 $k$**：抬高 $z_k$ 会**压低**你实际选的那个动作，梯度 $=-\pi(k)<0$

第二条是 **softmax 的零和性质**（分母共享，抬一个必压其余），也正是「等大反号」的来源。

**闭环验证**：用这个公式能直接推出 score function 恒等式，和「概率归一化求导」那条路殊途同归：

$$\sum_a\pi(a)\big(\mathbb{1}[k{=}a]-\pi(k)\big)=\underbrace{\sum_a\pi(a)\mathbb{1}[k{=}a]}_{=\ \pi(k)}-\pi(k)\underbrace{\sum_a\pi(a)}_{=\ 1}=\pi(k)-\pi(k)=0$$

**和交叉熵的关系**：监督学习里 $\dfrac{\partial(-\log\pi(y))}{\partial z_k}=\pi(k)-\mathbb{1}[k{=}y]$，
即经典的「**softmax 减 one-hot**」—— 和上式只差一个负号（交叉熵是负对数似然）。
所以 **PG 的 actor loss 和监督学习的交叉熵梯度公式是同一个**，
区别只在前面乘的系数：监督学习恒为 1，策略梯度是 advantage。
（`policies.py` 模块 docstring 里「PG 就是加权的交叉熵」，根子在这。）

**代回本例**：在 50/50（$\pi=(0.5,0.5)$）处对 $z_0$ 求导，torch 实测 `d logπ/d z = [0.5, -0.5]`：

$$s(0)=1-0.5=+0.5,\qquad s(1)=0-0.5=-0.5$$

**真实值是 $\pm 0.5$**，表格里取 $\pm 1$ 是整体放大 2 倍，纯粹为了算术干净。
score 线性进入估计量，放大 $c$ 倍会让期望乘 $c$、方差乘 $c^2$ —— **方差之比不变**：

| $c$ | $\mathbb{E}[\hat g_{\text{traj}}]$ | $\mathbb{E}[\hat g_{\text{rtg}}]$ | $\operatorname{Var}$[traj] | $\operatorname{Var}$[rtg] | 方差比 |
|---|---|---|---|---|---|
| 1/2（真实值） | 2.750 | 2.750 | 22.69 | 7.69 | **2.9512** |
| 1（本文用） | 5.500 | 5.500 | 90.75 | 30.75 | **2.9512** |
| 10 | 55.0 | 55.0 | 9075 | 3075 | **2.9512** |

**为什么必然是「等大反号」** —— 这不是我为了让例子好看而设的，
恰恰是本文要演示的那个恒等式**逼出来**的：

$$\mathbb{E}_{a\sim\pi}[s(a)]=0.5\,s(0)+0.5\,s(1)=0\;\Longrightarrow\;s(1)=-s(0)$$

所以后面「$X_1$ 的 $+10$ 与 $-10$ 恰好抵消」不是巧合，是概率归一化的必然结果。

> **只有 50/50 时才等大。** 一般地 $\pi(0)=p$ 时 $s(0)=1-p$、$s(1)=-p$
> （torch 实测 $p=0.731$ 时为 $+0.269$ / $-0.731$），大小不同，
> 靠**加权**抵消：$p(1-p)+(1-p)(-p)=0$。等大只是 $p=0.5$ 的特例。

</details>

**两个估计量**（单条轨迹）：

$$\hat g_{\text{traj}}=s(a_0)\cdot R+s(a_1)\cdot R,\qquad R=r_0+r_1$$
$$\hat g_{\text{rtg}}=s(a_0)\cdot(r_0+r_1)+s(a_1)\cdot r_1$$

差值就是第 1 步的「过去奖励」项：$X_1=s(a_1)\cdot r_0$。
（第 0 步没有过去，$P_0=0$，所以只有一项差异。）

### 全部四条轨迹，每条概率 1/4

| $a_0$ | $a_1$ | $r_0$ | $r_1$ | $R$ | $s(a_0)$ | $s(a_1)$ | $\hat g_{\text{traj}}$ | $\hat g_{\text{rtg}}$ | $X_1$ |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 10 | 1 | 11 | $+1$ | $+1$ | **22** | **12** | $+10$ |
| 0 | 1 | 10 | 0 | 10 | $+1$ | $-1$ | **0** | **10** | $-10$ |
| 1 | 0 | 0 | 1 | 1 | $-1$ | $+1$ | **0** | **0** | $0$ |
| 1 | 1 | 0 | 0 | 0 | $-1$ | $-1$ | **0** | **0** | $0$ |

### 结果

| | 期望 | 方差 | 标准差 |
|---|---|---|---|
| trajectory-centric | **5.50** | 90.75 | 9.53 |
| reward-to-go | **5.50** | 30.75 | 5.55 |
| 差值 $X_1$（过去项） | **0.00** | 50.00 | 7.07 |

- 期望 $11/2$ 对 $11/2$，**精确相等**，不是近似
- 方差 $90.75/30.75=2.95$ 倍
- $X_1$ 自己的标准差 7.07 **比信号本身 5.50 还大**

---

## 逐行读懂：$X_1$ 在干什么

看前两行 —— 它们的**过去完全一样**（都是 $a_0=0$，拿到 $r_0=10$），只有第 1 步的动作不同：

| $a_0$ | $a_1$ | $r_0$ | $X_1$ |
|---|---|---|---|
| 0 | **0** | 10 | $\mathbf{+10}$ |
| 0 | **1** | 10 | $\mathbf{-10}$ |

**同样已经到手的 10 分，只因为第 1 步选了不同动作，这一项就翻符号。**

而 $r_0$ 在第 0 步就定死了，跟 $a_1$ 选什么**毫无关系**。所以 $X_1$ 在做的事是：

> **拿一笔与 $a_1$ 无关的钱，随机地奖励或惩罚 $a_1$。**

$a_1$ 是好是坏跟 $r_0$ 无关，所以这个奖惩纯属噪声。

**为什么期望是 0**：因为 $a_1$ 是 50/50，而 $r_0$ 不依赖 $a_1$ ——
$+10$ 和 $-10$ 出现的概率相等，**配对相消**。

**注意措辞**：不是「这一项消失了」，是「**它的平均值是 0**」。
每一次采样它都实实在在地取到 $\pm 10$，从来不是 0。
$\operatorname{Var}(X)=\mathbb{E}[X^2]-(\mathbb{E}[X])^2$ 里，$\mathbb{E}[X]=0$
只消掉了第二项，$\mathbb{E}[X^2]$ 照样是正的。

> **抛硬币赢 1 输 1**：期望收益 0，但你每一把都在真金白银地赢或输。

---

## 「方差大 = 难收敛」，两个机制

### 机制一：更费样本（监督学习里也一样）

$N$ 个独立样本的均值，方差按 $N$ 缩小：

$$\operatorname{Var}(\bar g_N)=\frac{1}{N^2}\operatorname{Var}\!\Big(\sum_{i=1}^N \hat g_i\Big)=\frac{1}{N^2}\cdot N\sigma^2=\frac{\sigma^2}{N}$$

（用到两条：独立变量之和方差相加；$\operatorname{Var}(cX)=c^2\operatorname{Var}(X)$。）

令两个估计量达到**相同精度**，即 $\operatorname{Var}(\bar g)$ 相等：

$$\frac{\sigma_A^2}{N_A}=\frac{\sigma_B^2}{N_B}\quad\Longrightarrow\quad \frac{N_A}{N_B}=\frac{\sigma_A^2}{\sigma_B^2}$$

**样本量之比 = 方差之比，线性。** 玩具例子上验证过：方差比 2.951，
要让 $\text{std}(\bar g)\le 0.30$ 分别需要 1009 和 342 个样本，比值 2.950 ✓

> **别把标准差比和方差比搞混** —— 差一个平方：
>
> | 问的问题 | 答案 |
> |---|---|
> | 给定 $N$，噪声（标准差）多大 | $\sigma/\sqrt N$，**开方** |
> | 要达到给定精度，需要多少 $N$ | $N\propto\sigma^2$，**线性于方差** |

### ⚠️ 这条公式**不能**用来预测「跑多少步能到 200」

一度想拿玩具例子的方差比去解释实验里的步数比，这是错的 —— 那是**两个不同环境、两种不同量**。
CartPole 上实测的真实方差比（200 个独立 batch 估计，训练不同阶段各测一次）：

| 训练迭代数 | 平均轨迹长度 | traj 总方差 | rtg 总方差 | 方差比 |
|---|---|---|---|---|
| 0 | 19.5 | 0.055 | 0.0097 | 5.66 |
| 20 | 133.2 | 123.3 | 10.36 | 11.90 |
| 50 | 154.0 | 102.9 | 21.35 | 4.82 |
| 100 | 167.0 | 339.9 | 42.40 | 8.02 |

**方差比在 5–12 之间跳，根本不是常数**，更不是实验里那个 3.0。

$N\propto\sigma^2$ 来自 SGD 收敛率分析，它的前提在 RL 里全都破了：

| 前提 | RL 里 |
|---|---|
| 目标函数固定 | ❌ 策略变 ⇒ 采样分布变 ⇒ 目标在动 |
| 方差是常数 | ❌ 见上表 |
| 衡量「梯度精度」 | ❌ 「首达 200」是噪声过程的**阈值穿越时刻**，不是同一个量 |
| 多 seed 取平均 | ❌ 单 seed；同代码同 seed 在 GPU/CPU 上跑出过 62.71 vs 200 |

**能说的**：方差大 ⇒ 更新更抖 ⇒ 更费样本，**方向确定**。
**不能说的**：从方差比推出步数比。

### 上表真正重要的不是比值，是绝对值

```
traj 总方差:  0.055  →  339.9      训练过程中涨了约 6000 倍
rtg  总方差:  0.0097 →   42.4      涨了约 4400 倍
```

**随着策略变好，梯度方差暴涨三四个数量级。** 这正是 $P_t=t$ 论证的预言：
轨迹从平均 19.5 步长到 167 步，回报从 ~20 涨到 ~167，方差大致正比于回报的平方。

**这是崩塌发生在训练中后期的直接证据**：开局梯度很干净（方差 0.055），
策略变好后方差变成 340，学习率却没变 —— 等效步长跟着暴涨，于是一脚踩空。

也正面解释了 `-na` 为什么是四个开关里效果最猛的：把 advantage 尺度钉死在 std=1，
直接掐断「策略变好 ⇒ 方差暴涨 ⇒ 崩塌」这条链。

### 机制二：RL 特有的恶性循环 —— 这个才是崩塌的原因

**监督学习**里方差大只是「慢」：走错一步，数据集还在那儿，下一步照样纠正回来。

**RL 里数据是策略自己采的**：

```
梯度噪声大  →  某一步把策略推歪  →  用歪掉的策略去采样
                   ↑                        ↓
                   └──── 采到的数据更差 ←────┘
```

一旦进入这个循环，不是「慢一点」而是**塌下去**。

`REPORT.md` 图 2 里 `cartpole_lb`（无 rtg 无 na）在 20–30 万步之间掉到 71.33，
花了约 10 万步才爬回来 —— 那不是噪声抖动，是这个反馈环。

**这就是为什么 RL 里方差控制的优先级远高于监督学习。**

---

## 本作业的四种技巧，都在解这一个问题

| PDF 节 | 技巧 | 做法 | 有偏吗 | 阶段 |
|---|---|---|---|---|
| §2.2.1 | **reward-to-go** | 删掉与 $a_t$ 无关的过去奖励项 | **无偏** | 1 |
| §2.2.3 | **baseline** $V_\phi(s)$ | 减掉只依赖 $s_t$、与 $a_t$ 无关的量 | **无偏** | 2 |
| §2.3 | **advantage normalization** | $(A-\mu)/(\sigma+\varepsilon)$ | 技术上**有偏**（$\mu$ 依赖采样到的这批数据） | 1 |
| §2.2.4 | **GAE** | 用 $V_\phi$ 替代部分 Monte Carlo 回报 | **有偏**（$\lambda<1$ 时），偏差换方差 | 3 |

前两种是**纯赚**，靠的是同一个 score function 恒等式：
只要乘在 $\nabla_\theta\log\pi_\theta(a_t|s_t)$ 上的那个量**与 $a_t$ 无关**，它的期望贡献就是零。

- reward-to-go：那个量是 $P_t$（过去奖励）
- baseline：那个量是 $b=V_\phi(s_t)$

**同一个恒等式，两处用法。** 搞懂一次，阶段 2 的「为什么减 baseline 不引入偏差」自动就通了
（PDF 式 12 下面那行 $\nabla_\theta\mathbb{E}[b]=\mathbb{E}[\nabla_\theta\log\pi_\theta\cdot b]=0$）。

后两种是**主动用偏差换方差** —— PDF §2.3 明说 normalization "is technically biased"，
§2.2.4 明说用 $V_\phi$ "comes at the cost of introducing bias"。
这是 bias-variance tradeoff，不是漏洞。

> 往后看：hw3 的 target network、final project 的 PPO clip / GRPO 的组内归一化，
> 全都是这张表的延伸。**「哪些量与当前动作无关」是判断能不能白拿的唯一标准。**

---

## 可复现

```python
from itertools import product
from fractions import Fraction as F

r0 = lambda a: 10 if a == 0 else 0
r1 = lambda a: 1  if a == 0 else 0
s  = lambda a: 1  if a == 0 else -1          # score ∇log π(a)

gt, gr, X1 = [], [], []
for a0, a1 in product([0, 1], [0, 1]):
    R = r0(a0) + r1(a1)
    gt.append(s(a0) * R + s(a1) * R)                          # trajectory-centric
    gr.append(s(a0) * (r0(a0) + r1(a1)) + s(a1) * r1(a1))     # reward-to-go
    X1.append(s(a1) * r0(a0))                                 # 差值：过去奖励项

mean = lambda v: sum(map(F, v)) / len(v)
var  = lambda v: sum((F(x) - mean(v)) ** 2 for x in v) / len(v)
for nm, v in [("traj", gt), ("rtg", gr), ("X1", X1)]:
    print(f"{nm:5} 期望={float(mean(v)):6.2f}  方差={float(var(v)):6.2f}")
# traj  期望=  5.50  方差= 90.75
# rtg   期望=  5.50  方差= 30.75
# X1    期望=  0.00  方差= 50.00
```

用 `Fraction` 而不是 float，是为了让「期望精确相等」这件事没有浮点误差的余地。
