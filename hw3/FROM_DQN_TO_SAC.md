# 从 DQN 到 SAC：argmax 是怎么把路堵死的

> **这份记什么**：为什么值函数方法在连续动作空间上失效，以及 actor 是用什么机制绕过去的。
> 一条因果链 + 几组实测数字，不是算法综述。
>
> **不记什么**：
> - 「hw3 这次为什么这么写」→ `NOTES.md`
> - 「跑出来多少」→ `REPORT.md`
> - 「公式怎么翻译成代码」→ `../hw2/FORMULA_TO_CODE.md`
> - 「形状怎么想」→ `../hw2/SHAPES.md`
>
> 和那两份一样，这是**跨作业复用**的：hw4（model-based）、hw5（exploration）、
> final project 里只要碰连续控制，就是同一条链。

---

## 一、先破一个直觉：不是「网络架构」的问题

常见的误解是「DQN 的 critic 只吃 observation，所以处理不了连续动作」。**不对。**

把 Q 网络写成 `(s, a) → 标量` 完全可行，**即使动作是离散的**。代价只是效率：
要拿到所有动作的 Q 值得跑 n 次前向，而 `obs → (n 个 Q)` 一次就够。

两种写法在本仓库里同时存在，正好对照：

```python
# DQNCritic          (networks/critics.py)
forward(obs)          →  (B, num_actions)     每个动作一个输出神经元
# StateActionCritic   (networks/critics.py)
forward(obs, acs)     →  (B,)                 一个标量
```

`num_actions` 是**建网络时**的参数（决定输出层宽度），不是 forward 的输入 ——
动作不是被「喂进去」的，是被「排在输出轴上」的。

**离散动作有限个，所以能一人分一个格子。这才是 DQNCritic 那种写法的前提。**

---

## 二、真正的拦路虎：算法里那两处 max

DQN 有两个地方必须对动作取最大值：

```
get_action       贪心动作 = argmax_a Q(s, a)
update_critic    y = r + γ(1-d)·max_{a'} Q_θ̄(s', a')
```

- **离散**：枚举 n 个取最大，是**精确解**，一行 `argmax(dim=-1)`。
- **连续**：`max_a Q(s,a)` 是一个非线性优化问题，**没有闭式解**，
  而且要**每个状态、每个梯度步**重解一次。

PDF §3 开篇说的就是这件事：

> DQN ... requires you to be able to calculate max_a Q(s,a) **in closed form**.
> Doing this is trivial for discrete action spaces ... but in continuous action
> spaces this is potentially a **complex nonlinear optimization problem**.

**所以：架构可以换，argmax 换不掉。**

---

## 三、"那我采样一批动作取最大" —— 这是真技术，但会崩

这个想法不是异想天开，有一整支工作：

| 方法 | 怎么近似 argmax |
|---|---|
| **QT-Opt**（Google 2018，机器人抓取） | CEM（交叉熵方法）迭代采样 |
| **CAQL** | 转成混合整数规划求解 |
| **NAF** | 限制 Q 对 a 是二次型 ⇒ argmax 有闭式解 |

问题不是「能不能」，是「够不够好」。实测（构造一个在 a* 处取最大值 1.0 的 Q，
看随机采样 K 个动作能逼近到多少）：

| `ac_dim` | 环境 | K=64 | K=1024 | K=16384 |
|---|---|---|---|---|
| 1 | InvertedPendulum | **1.000** | 1.000 | 1.000 |
| 3 | Hopper | 0.969 | 0.987 | 0.995 |
| 6 | **HalfCheetah** | 0.017 | 0.617 | **0.635** |
| 17 | Humanoid | 0.000 | 0.000 | 0.001 |

**6 维时采 16384 个动作只够到真实最大值的 63%** —— 而这是每个状态、每个梯度步的代价。
17 维直接归零。

网格搜索更绝望（每维 20 格）：

```
ac_dim=1 :          20 个点
ac_dim=3 :       8,000
ac_dim=6 :  64,000,000      ← 每个梯度步都要算一次
ac_dim=17:  1.3 × 10^22
```

这就是维度诅咒的具体样子。**注意 ac_dim=1 那行全是 1.000** ——
一维时采样法完全可行，所以 InvertedPendulum 上任何做法都能work，
它是 sanity check 而不是难题。

---

## 四、SAC 的答案：把优化摊销进一个网络

actor `π_θ` 就是一个「记住了 argmax_a Q(s,a) 长什么样」的网络。

| 做法 | 每步代价 | 质量随维度 |
|---|---|---|
| 采样取最大 | K 次 critic 前向 | 崩塌 |
| CEM（QT-Opt） | 多轮 × K 次前向 | 好一些，仍贵 |
| **学 actor** | **1 次 actor 前向** | 训练中持续改进 |

而训练 actor 的方式（PDF §3.4 的重参数化）比「搜索」更聪明：

$$\nabla_\theta \mathbb{E}_{s\sim D,\epsilon\sim N}[Q(s, \mu_\theta(s)+\sigma_\theta(s)\epsilon)]$$

**不是去试哪个动作好，而是让梯度直接告诉你动作该往哪挪。**
梯度从 Q 反传到动作、再传到 θ，一步到位，不随维度崩塌。

这也解释了 SAC 的其余设计都是从哪来的：

| 组件 | 为了解决什么 |
|---|---|
| actor `π_θ` | 替代 argmax |
| 重参数化 `.rsample()` | 让梯度能穿过采样这一步 |
| 熵 bonus | 连续空间没有 ε-greedy，需要别的探索机制 |
| clipped double-Q | max 的乐观偏差还在，用两个 critic 取 min 压制 |

**四个组件，前两个直接冲着 argmax 去，后两个是替换 argmax 之后必须补的窟窿。**

---

## 五、一句话的判据

> **动作能枚举 → 值函数方法（DQN 家族）。不能枚举 → 必须显式学策略（actor-critic 家族）。**

「能不能枚举」而不是「离散还是连续」—— 一个 10^6 个离散动作的空间同样枚举不了
（例如组合优化、大词表的语言模型动作空间），那里也要用策略梯度或采样近似。
反过来，一维连续动作（InvertedPendulum）用采样法就够，上表第一行是证据。
