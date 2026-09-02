# HW3 实现笔记

> **这份记什么**：我怎么走到最终代码的 —— 写错的版本、什么现象让我发现错了、试过又放弃的写法。
>
> **不记什么**：
> - 「最终代码为什么长这样」→ 写进源文件 docstring，沿用 hw1/hw2 的三段式（*做的事 / 为什么 / 术语解释*）
> - 「跑了什么、结果多少」→ 写进 `REPORT.md`
> - 「数学公式怎么机械地翻译成 PyTorch 代码」→ `../hw2/FORMULA_TO_CODE.md`（跨作业复用，不 fork）
> - 「张量形状/轴怎么想、哪些形状错误不报错」→ `../hw2/SHAPES.md`（同上）
> - 「因子多到跑不完全因子时怎么设计消融」→ `../hw2/ABLATION.md`（§2.6 调参会用到）
> - 「为什么 DQN 到连续动作就失效、actor 怎么绕过 argmax」→ `FROM_DQN_TO_SAC.md`
>   （本次新增，同样跨作业复用 —— hw4/hw5/final project 碰连续控制都是这条链）
>
> 分界线：**读最终代码的人需要知道的** 留在代码里；**只有写代码的我经历过的** 留在这里。
>
> ⚠️ 待决定：`SHAPES.md` / `FORMULA_TO_CODE.md` / `VARIANCE.md` / `ABLATION.md` 四份都声明了
> 跨作业复用，现在物理上还在 `hw2/` 下。要不要提到仓库根目录？hw3 里遇到的新坑一律**续写
> hw2 那份**，不在 hw3 复制第二份（两份会分叉）。

---

## 阶段总览

| 阶段 | PDF 节 | 动的文件 | 验收标准 | commit | 状态 |
|---|---|---|---|---|---|
| 0. 环境 | §1 | 无代码改动 | 六个 env 都能 make/reset/step | — | ✅ |
| 1. 基础 DQN | §2.4 | `dqn_agent.py` `run_dqn.py` | CartPole-v1 训练中**至少一次** eval return = 500 | 本次 | ✅ 40 个评估点中 10 次 500 |
| 2. Double-Q | §2.5 | `dqn_agent.py` | LunarLander-v2 ≥ 200；MsPacman ≈ 1500 | 本次 | ✅ LunarLander 17 次≥200；MsPacman 最好 1924 |
| 3. 超参敏感性 | §2.6 | 新增 yaml（无源码改动） | LunarLander 4 组设置同图 | | ⬜ |
| 4. SAC 数据流 + bootstrapping | §3.1–3.2 | `run_sac.py` `sac_agent.py` | InvertedPendulum Q 值稳定（不发散、不恒零） | 本次 | ✅ q=5.06±0.27，闭式解预测 2.8~5.5 |
| 5. 熵 bonus | §3.3 | `sac_agent.py` | InvertedPendulum entropy → ≈ log 2 ≈ 0.69 | | ⬜ |
| 6. 重参数化 actor | §3.4 | `sac_agent.py` | InvertedPendulum ≈ 1000；HalfCheetah ≥ 6000 | | ⬜ |
| 7. 自动温度 | §3.5 | `sac_agent.py` | InvertedPendulum 仍 ≈ 1000；HalfCheetah autotune 对比图 | | ⬜ |
| 8. clipped double-Q | §3.6 | `sac_agent.py` | Hopper clipq ≥ 1500，与 singleq 对比 | | ⬜ |

> 沿用 hw2 的约定：**一个阶段一个 commit**，这样 `git log -p` 本身就是精确到行的实现过程记录。
> 阶段 4–8 全部动 `sac_agent.py` 同一个文件，重叠比 hw2 严重，更要靠 commit 切开。
>
> PDF §2.4/2.5 的验收都是「**至少一次**达到」，不是「稳定保持」—— 别看到曲线掉下来就以为是 bug。

---

## 实现策略

> 沿用 hw2 的判据：**TODO 在问「数据长什么样」→ 自顶向下；在问「这个数怎么算」→ 自底向上。**
> hw3 的 TODO 分布是否也是这个形状？写完阶段 1 回来填。

> TODO

---

## 阶段 0 — 环境（PDF §1）

`uv sync` 已完成。`swig`（`gym[box2d]` 编译需要）在 `~/.local/bin/swig`，同时 venv 里也有 `swig` 4.4.1 的 wheel —— 没去追是哪一个起的作用，反正 box2d 装成了。

### 验证记录

| 项 | 结果 |
|---|---|
| Python | 3.10.20 |
| torch / CUDA | 2.10.0，CUDA 可用，RTX 4090 |
| gym / mujoco / box2d-py | 0.25.2 / 2.2.0 / 2.3.5 |
| numpy | 1.26.4（`<2.0`，gym 0.25.2 要求） |
| 六个 env `make`+`reset`+`step` | CartPole-v1 ✅ LunarLander-v2 ✅ InvertedPendulum-v4 ✅ HalfCheetah-v4 ✅ Hopper-v4 ✅ MsPacmanNoFrameskip-v4 ✅（Atari ROM 已装，ALE 0.7.5） |
| `scripts.run_dqn` / `run_sac` / `agents.*` / `configs.*` import | ✅ |

**有 4090 ⇒ MsPacman 和 HalfCheetah 可以本地跑，Modal 不是必需的。**

跑起来会刷一堆 `DeprecationWarning`（gym 未维护 / old step API / `np.bool8`）——
starter code 本来就按 old step API 写（`run_dqn.py` 里 `env.step()` 解包成四元组），是常态，不用管。

---

## 阶段 1 — 基础 DQN（PDF §2.4）

跑通目标：CartPole-v1 训练中至少一次 eval return 到 500。此阶段 **`use_double_q` 走 `False` 分支**，
double-Q 留到阶段 2。

### 前置阅读

**PDF §2.2 原文点名的，只有两个**（原话："you should start by reading the following files thoroughly"）：

- [x] `src/configs/dqn_config.py` —— `basic_dqn_config()` 返回的 dict 的结构 ✅ 见「接口速查」
- [x] `src/infrastructure/replay_buffer.py` —— `ReplayBuffer.sample()` 返回什么 ✅ 见「接口速查」
      （`MemoryEfficientReplayBuffer` 是 Atari 才用的，本阶段可跳过）

> §2.2 同时说明了**要实现的**是 `src/agents/dqn_agent.py` 和 `src/scripts/run_dqn.py`。

**下面两项 PDF 没点名，是我自己加的**：

- [x] `src/networks/critics.py` `DQNCritic.forward` 的输入输出形状 ✅ `(B, *obs_shape) -> (B, num_actions)`
      —— 加它是因为写 `update_critic` 时要靠这个形状推 `gather` 那一步
- [x] PDF §2.3（不是 §2.2）列的三个已实现 trick：exploration schedule / lr schedule / grad clipping ✅ 见接口速查的两 config 对比表
      —— 不用写，但要能在 `dqn_config.py` 里指出它们各自的实体

### 接口速查

> **只记事实**（键名、形状、dtype、谁传给谁），不记感想。超过 20 行就是在写理解而不是事实了。
> 写 `update_critic` 时会反复查这一节，省得跳三个文件。

**`basic_dqn_config()` 返回的 dict —— 分界不是「网络 vs 环境」，是「谁消费它」**

| 键 | 消费者 | 内容 |
|---|---|---|
| `agent_kwargs` | `DQNAgent.__init__`（`run_dqn.py` 里 `**` 展开） | `make_critic` `make_optimizer` `make_lr_schedule` `discount` `target_update_period` `clip_grad_norm` `use_double_q` |
| `exploration_schedule` | 训练循环 `epsilon = .value(step)` | agent **看不到**；ε 是循环算好传进 `get_action(obs, epsilon)` 的 |
| `total_steps` `batch_size` `learning_starts` | 训练循环 | 循环长度 / 每次采样条数 / 第几步开始训练 |
| `make_env` `log_name` | `run_dqn.py` 建 env 和 logger | |

⚠️ **写 `dqn_agent.py` 时手里只有 `agent_kwargs` 那七项**，其余都够不着。

**两个 config 同名字段行为不同**

| | `dqn_basic` | `dqn_atari` |
|---|---|---|
| critic | `DQNCritic`（MLP） | CNN，前置 `PreprocessAtari` 除 255 |
| lr schedule | `ConstantLR(factor=1.0)` —— **不衰减，是占位** | `LambdaLR`+Piecewise，后半程 0.5× |
| ε schedule | 1.0 → 0.1(30%) → 0.02(60%) | 1.0 保持到 20k → 0.01(50%) |
| `clip_grad_norm` | `None` | `10.0` |

**`ReplayBuffer.sample(B)` 返回 dict**（实测，B=3、obs 4 维）

| 键 | shape | numpy dtype | 过 `ptu.from_numpy` 之后 |
|---|---|---|---|
| `observations` | `(B, *obs_shape)` | float32 | `torch.float32` |
| `actions` | `(B,)` | int64 | `torch.int64` |
| `rewards` | `(B,)` | float64 | `torch.float32`（`from_numpy` 把 float64 降成 float32） |
| `next_observations` | `(B, *obs_shape)` | float32 | `torch.float32` |
| `dones` | `(B,)` | bool | `torch.bool` |

> dtype **不是写死的**，是第一次 `insert` 时按传进来的数据 `np.empty` 定的。

⚠️ **键名是复数，`DQNAgent.update` 的形参是单数**（`obs` `action` `reward` `next_obs` `done`）—— 对不上。

⚠️ **`dones` 是 `torch.bool`，`1 - dones` 直接抛 `RuntimeError`**
（"Subtraction, the `-` operator, with a bool tensor is not supported"）。

**采样机制**：`np.random.randint(low, high, size)` —— 前两个是区间，`size` 是掷几次，
返回**长度 B 的下标数组**，可重复 ⇒ 有放回均匀采样。再用 fancy indexing
（`arr[idx]`，第一维变成 B）一次取出整个 batch。
`% self.max_size` 是因为 `self.size` 只增不减、绕圈后会超界。
（副作用：绕圈后采样不再严格均匀。hw3 容量 1e6 ≥ `total_steps`，绕不到。）

**`MemoryEfficientReplayBuffer`（阶段 2 MsPacman 才用）**

`run_dqn.py` 按观测维度二选一：`len(obs_space.shape) == 3` → 省内存版；`== 1` → 普通版。
所以 **CartPole 走的是普通 `ReplayBuffer`**。

思想：相邻观测重叠 3 帧，每帧最多被存 8 次（obs×4 + next_obs×4）。
1e6 条 transition：朴素 **52.6 GiB** → framebuffer+索引 **13.2 GiB**（4×）。
做法是 `framebuffer` 只存单帧，每条 transition 存 4 个 int64 帧号，
`sample()` 因此是**两层索引**：`rand_indices` → 取出 4 个帧号 → 再去 `framebuffer` 取像素，拼成 `(B, 4, 84, 84)`。
`_compute_frame_history_idcs` 的 `np.maximum(..., trajectory_begin)` = episode 开头凑不满 4 帧时
重复第一帧、不借上一个 episode；`on_reset` 就是用来记这个边界的。


### TODO 清单

> 行号会漂移，按函数名找。源码里的标记是 `TODO(Section 2.4)` … `# ENDTODO`。

- [x] `dqn_agent.py` `get_action` —— ε-greedy ✅ 已验证（ε=0/0.5/1 三档）
- [x] `dqn_agent.py` `update_critic` —— target 值 ✅ 手算用例验证
- [x] `dqn_agent.py` `update_critic` —— 预测值与 loss ✅ 固定 target 收敛测试
- [x] `dqn_agent.py` `update` —— 调 critic 更新 + 按 `target_update_period` 更新 target ✅
- [x] `run_dqn.py` 训练循环 —— 取 action ✅
- [x] `run_dqn.py` 训练循环 —— 从 replay buffer 采样 `config["batch_size"]` 条 ✅
- [x] `run_dqn.py` 训练循环 —— 调 `agent.update` ✅

### 关键决定

**ε-greedy 用讲义口径（排除贪心动作），不用常见实现口径**

课程材料自己不一致：讲义写 π(a|s) = 1-ε 若 a=argmax，否则 **ε/(|A|-1)**；
而多数参考实现是「以 ε 的概率从**全部** n 个动作里均匀抽」，即 P(贪心)=1-ε+ε/n。
两者其实是同一族，只差 ε 的缩放（ε₁ = ε₂·(n-1)/n）—— 不是对错问题。选讲义口径是为了和课程口径一致。

⚠️ **代价（已量化，接受）**：ε₁=1 落在实现式表达不了的区域（需要 ε₂ = n/(n-1) > 1），
比均匀随机更「反贪心」。`cartpole.yaml` 的 schedule 恰好从 1.0 起步，且 `learning_starts=1000`
之前 critic 是冻结的随机网络：

| step | ε | 讲义式 P(贪心) | 实现式 P(贪心) |
|---|---|---|---|
| 0–100 | ≈1.00 | **≈0.00** | 0.50 |
| 1000 | 0.97 | 0.03 | 0.515 |
| 20000 | 0.40 | 0.60 | 0.80 |
| 30000 | 0.10 | 0.90 | 0.95 |

即前 ~1000 步近乎确定性地执行「随机网络的 argmin」，填进 buffer 的数据相关性偏高。
占 100000 步里的 1%，且 CartPole 杆倒得快、频繁 reset 仍有初始状态随机性 —— 判断可接受。
**若 CartPole 迟迟不收敛，这里是第一个要回来看的地方。**

次要代价：讲义式**两支都必须先做一次前向**（要排除贪心动作就得先知道它是谁），
实现式在探索时可以完全跳过网络。CartPole 无所谓，Atari 上每个探索步白跑一次 CNN。

**实现细节**：排除贪心动作用「贪心下标 + `[1, n)` 随机偏移，再对 n 取模」，
一次抽中且均匀，不需要拒绝采样（`while` 重试）。

### 验证记录

> 验收命令：
> ```
> uv run src/scripts/run_dqn.py -cfg experiments/dqn/cartpole.yaml --eval_interval 2500
> ```

**`get_action` 单测**（不跑训练，直接造 agent 喂固定 obs，各 12000 次采样）：

| n | ε=0 | ε=0.5 | ε=1.0 |
|---|---|---|---|
| 2 | P(贪心)=1.000 | 0.498（理论 0.500） | **0.000**，取值恒为 `[1]` |
| 3 | 1.000 | 0.498 / 非贪心各 0.246~0.256（理论 0.250） | 0.000，取值 `[1,2]` |
| 9 | 1.000 | 0.498 / 各 0.060~0.065（理论 0.0625） | 0.000，取值 `[0..8]` **缺 5**=贪心 |

n=9 那行取值范围缺了贪心下标，是「排除」真的生效的直接证据。
另外确认：不传 `epsilon` 时走默认 0.0（eval 路径，`utils.py:37`），返回类型是 Python `int`。

**`update_critic` 单测**（把源码块抽出来 exec，喂手算输入）：

| 输入 | 期望 | 实测 |
|---|---|---|
| `qa=[[1,5,2],[9,0,3]]` | `next_action_B = [1,0]` | ✅ |
| 同上 | `next_q_values_B = [5,9]` | ✅ |
| `r=[1,2]`, γ=0.99, `done=[F,T]` | `target = [1+0.99·5, 2+0]` = `[5.95, 2.0]` | ✅ |

第 2 行 `done=True` → target 退化成纯 reward，bootstrap 项确实被掐掉。

**梯度能不能流**（`done=True` 让 target 恒为 `reward=5.0`，连跑 300 步）：

| step | loss | q_values |
|---|---|---|
| 0 | 23.087 | 0.197 |
| 100 | 0.134 | 5.056 |
| 200 | 0.011 | 4.997 |
| 299 | 0.00032 | 5.000 |

q 值收敛到 5.0、loss → 0 ⇒ 梯度流通且方向正确。

**CartPole 正式验收** —— `exp/CartPole-v1_dqn_sd1_20260825_151358`

```
uv run src/scripts/run_dqn.py -cfg experiments/dqn/cartpole.yaml --eval_interval 2500 --no_gpu
```

| 指标 | 结果 |
|---|---|
| 40 个评估点中 `Eval_AverageReturn = 500` | **10 次** |
| 首次达到 500 | **step 10000** |
| step ≥ 50000 的 20 个评估点 | 均值 370.4，其中 9 次满分 |
| 最终评估点（97500） | 500.0 |

PDF 要求「训练中**至少一次**到 500」，达标。

⚠️ **曲线中段会大幅回落**（15000–50000 掉到 80~200，末段 90000–95000 也掉到 ~100），
这是 DQN 常态不是 bug：ε 那时还在 0.4 以上、buffer 里旧策略数据占比高、
target 每 1000 步才同步。**别看到掉下来就以为写错了** —— 验收标准写的是「至少一次」。

**GPU vs CPU 实测（20000 步）**

| | it/s |
|---|---|
| CPU (`--no_gpu`) | **1038** |
| GPU (RTX 4090) | 871 |

CPU 快 19%。hw2 的结论成立但没那么夸张（hw2 是 1.7×）—— DQN 每步都做一次
batch=128 的梯度更新，GPU 的活比 hw2 的 PG 多，差距被拉近。瓶颈仍是 `env.step()`
和逐步 `get_action` 的 host→device 拷贝。**CartPole 用 `--no_gpu`**；
LunarLander 网络宽 4 倍（256）届时重测；MsPacman 是 CNN，必然用 GPU。

> TODO

### 踩的坑

**① `1 - done` 直接抛 RuntimeError** —— `done` 是 `torch.bool`，torch 不支持 bool 张量做减法。
必须 `1 - done.float()`。（接口速查里记过，写的时候还是撞上了。）

**② `.max(dim=-1)` 返回的不是张量** —— 是 namedtuple `(values, indices)`，`.shape` 会
`AttributeError`。要纯张量得用 `.values` / `amax` / `argmax`。
但这里根本不该用 `max`：starter code 要「先 argmax 拿下标、再按下标取值」两步，
`max` 一步到位会跳过 `next_action`，阶段 2 的 double-Q 就没法只改一行。

**③ `squeeze` 变不出标量** —— 卡了三轮。`squeeze`/`unsqueeze` 只增删**长度为 1** 的轴，
元素个数永远不变，属于「改形状」；把 B 个数收成 1 个数是**归约**（`mean`/`sum`）。
`loss.backward()` 要 0 维标量，所以必须用 `self.critic_loss`（= `nn.MSELoss()`，
它做的正是「平方 + 对 B 取平均」两步）。逐样本误差 `(B,)` 不是损失。

**④ 漏 `.squeeze(1)` 会静默广播** —— `gather` 出来是 `(B,1)`，若不 squeeze，
`reward + γ(1-d) * (B,1)` 会广播成 **`(B,B)`**，不报错。抓住它的是 starter code 那条
`assert target_values.shape == (batch_size,)` —— 位置卡得精准，别删。
（右对齐、左补 1：`(B,)` → `(1,B)`，与 `(B,1)` 一碰就是 `(B,B)`。）

**⑤ T1.5 改名漏了下游引用** —— 把 `target_values` 改成 `target_values_B` 后，
方法末尾返回字典里的 `target_values.mean()` 没跟着改，`NameError`。
改名后 `grep` 一遍旧名字。

**⑥ `torch.randint` 默认建在 CPU 上** —— `get_action` 里 `greedy_action` 在 GPU，
`offset` 在 CPU，相加抛 `Expected all tensors to be on the same device`。
要传 `device=greedy_action.device`。
**单测时我用的是 `use_gpu=False`，纯 CPU 跑不出这个 bug** —— 冒烟测试才抓到。
教训：设备相关的 bug，CPU 单测一律看不见。

**⑦ `run_dqn.py` 里调错了方法** —— 一度写成 `agent.update_critic(...)`，绕过了
`agent.update(...)`。后果不是报错而是**静默失效**：target 网络永远停在随机初始化状态，
课件第 5 步从未执行。`update_critic` 的签名没有 `step`，这是个提示。

**⑧ `get_action` 忘了传 `epsilon`** —— 走默认 `0.0` ⇒ 全程纯贪心、零探索。
`exploration_schedule` 和讲义式 ε-greedy 全部白写。
判据：`run_dqn.py` 里 `epsilon = exploration_schedule.value(step)` 算出来的变量
如果没有任何地方用到，就是漏传了。

**⑨ `WANDB_MODE=disabled` 在这份代码里不生效** —— `run_dqn.py:224` 显式传了
`mode="online"` 给 `wandb.init()`，**显式参数优先级高于环境变量**。
和 hw2 不同（hw2 的脚本里 `WANDB_MODE=disabled` 是管用的）。
所以每次冒烟测试都会在 W&B 上留一个 run，得手动清理。

### 遗留疑问

- `get_action` 没加 `torch.no_grad()`。不加不会错（`ptu.to_numpy` 有 `.detach()`），
  只是每个环境步白建一次计算图。CartPole 无所谓，MsPacman 跑 1e6 步时回来评估。
- 字母表已为 hw3 扩了一个：hw2 的表写着「离散动作没有 `A`」，但 DQN critic 的输出
  **有**一根真正的动作轴。现采用 `A` = 动作轴长度（DQN 下 = `num_actions`，SAC 下 = `ac_dim`），
  代码里已在用（`next_qa_values_BA`）。**待办：回填进 `../hw2/SHAPES.md` 的字母表**，
  否则那份文档和实际用法就对不上了。

---

## 阶段 2 — Double-Q（PDF §2.5）

### TODO 清单

- [x] `dqn_agent.py` `update_critic` 的 `if self.use_double_q:` 分支 ✅ 手算用例验证
      —— 只改了 `next_action_B` 那一行，`next_q_values_B` / `target_values_B` 原样未动。
      当初把 target 那块拆成四步，就是为了这一刻只需改一行。

### 关键决定

**在线网络的输出内联，不落变量**

```python
next_action_B = self.critic(next_obs).argmax(dim=-1)
```

一度写成先存 `original_next_qa_values_BA = self.critic(next_obs)` 再 argmax。内联的理由：
它只用一次，落变量反而要给它起名，而任何带 `next_qa_values` 字样的名字都会和
target 那个混淆 —— 见下面「踩的坑 ①」，那个 bug 正是这么来的。

### 验证记录

> 验收命令：
> ```
> uv run src/scripts/run_dqn.py -cfg experiments/dqn/lunarlander.yaml     # 期望 ≥ 200，GPU，约 17 分钟
> uv run src/scripts/run_dqn.py -cfg experiments/dqn/mspacman.yaml        # 期望 ≈ 1500，约 3h
> ```
> `lunarlander.yaml` 里已经是 `use_double_q: true`。

**双分支手算用例**（把源码块抽出来 exec，两个网络喂固定值）

```
target Q_φ'(s',·) = [[1,5,2], [4,0,1]]
online Q_φ (s',·) = [[9,3,4], [0,8,2]]
```

| `use_double_q` | `next_action_B` | `next_q_values_B` | 手算 |
|---|---|---|---|
| `False` | `[1, 0]` | `[5.0, 4.0]` | target 自己 argmax，取 target 的值 ✅ |
| `True` | `[0, 1]` | `[1.0, 0.0]` | online argmax，取 **target** 的值 ✅ |

第 0 行最有鉴别力：online 挑了动作 0（它自己给 9.0），但取回来的是 `target[0]=1.0`。
**这个「悲观」就是消除高估偏差的全部机制** —— 在线网络看好的动作，
在 target 眼里可能很差，取 target 的值就不会把 max 的乐观偏差传下去。

**LunarLander 正式验收** —— `exp/LunarLander-v2_dqn_sd1_20260825_153744`

| 指标 | 结果 |
|---|---|
| 50 个评估点中 `Eval_AverageReturn ≥ 200` | **17 次** |
| 首次达标 | step **260,000**（241.5） |
| 最好 / 最终 | **274.0** / 224.7 |
| step ≥ 300,000 的 20 个评估点 | 14 个 ≥200 |

PDF 要求「至少一次 ≥200」，达标且非偶然。图见 `report/stage2_lunarlander.png`。

**GPU vs CPU（20000 步基准，`setup_wandb` 打桩以免污染 W&B）**

| | CartPole（hidden 64，4.6K 参数） | LunarLander（hidden 256，69K 参数） |
|---|---|---|
| CPU | **1038** it/s | 369 it/s |
| GPU | 871 it/s | **493** it/s |
| 结论 | CPU 快 19% | **GPU 快 34%** |

账在网络规模上翻转：15 倍参数量让矩阵乘终于压过 host→device 拷贝。
**LunarLander 不要加 `--no_gpu`。**

**为什么 GPU 利用率只有 14~20%**（`nvidia-smi dmon` 实测）

显存 490 MiB 里几乎没有模型：critic 270 KB、加 target 和 Adam 两个动量约 1 MB，
其余全是 CUDA context（驱动 + cuBLAS/cuDNN kernel），是碰 CUDA 的固定开销。
利用率低是因为**延迟受限而非吞吐受限** —— 每步里 `env.step()`（Box2D，CPU）、
`get_action` 的单样本 PCIe 往返、`replay_buffer.sample` 的 numpy 索引都在 CPU 上，
GPU 只在 `update` 那几个 64×256 的矩阵乘上干几十微秒。不用管，20% 的 4090 仍比 CPU 快 34%。

**MsPacman 会完全不同**（算过的账）：

| | 参数量 | 每样本 MACs | batch | 每次更新 MACs |
|---|---|---|---|---|
| LunarLander MLP | 69,124 | 68,608 | 64 | 4.4 M |
| MsPacman CNN | 1,688,745 | 9,347,584 | 32 | **299 M（68×）** |

单样本算力 136×。传输量涨得更凶（2 KB → 1764 KB，880×），但 1.7 MB 走 PCIe 4.0
只要约 70 μs，而 299M MACs 即使只发挥 10% 也是几百 μs —— **算力这次是主项**。
（Atari 存 uint8 而非 float32，传输已省 4 倍；除以 255 放在 GPU 上做，
就是 `PreprocessAtari` 那一层的意义。）
所以 MsPacman 必须用 GPU，CPU 跑不动。

⚠️ **但「GPU 利用率会显著提高」这个预测是错的** —— 实测 `nvidia-smi dmon` 仍是
0~32%、均值约 15%，和 LunarLander 差不多。算漏的一步：299M MACs 在 4090
（~80 TFLOPS fp32）上只有 **~7.5 μs** 纯计算，基数太小，乘 68 倍还是微不足道。
瓶颈只是**换了个 CPU 环节**，没有离开 CPU：ALE 模拟、帧预处理（灰度 + 缩放到 84×84）、
以及 `MemoryEfficientReplayBuffer.sample` 用两层 fancy indexing 拼 `(32,4,84,84)` uint8
（每次约 900 KB 的 numpy gather）。

显存同样印证：540 MiB vs LunarLander 的 490 MiB —— **参数量 24 倍，显存只多 10%**。
1.69M 参数 ×4 字节 ×4 份（网络 + target + Adam 两个动量）≈ 27 MB，加卷积激活约 50 MB，
其余全是 CUDA context。replay buffer 在 CPU 内存里，不占显存。

**教训：估 GPU 负载要看绝对耗时，不能只看相对倍数。** 68× 一个微不足道的量
仍然微不足道。判据应是「单次更新的 GPU 计算时间是否可与 env.step + 数据准备相比」。

实测速率 **≈321 it/s**，100 万步约 **52 分钟** —— 比 PDF 说的「约 3 小时」快得多，
那个估计应该是按较弱硬件给的。

### 踩的坑

**① 覆盖 `next_qa_values_BA` 会把 double-Q 变成比 vanilla 更糟的东西**

一度写成：

```python
if self.use_double_q:
    next_qa_values_BA = self.critic(next_obs)          # ← 覆盖了 target 的输出
    next_action_B = next_qa_values_BA.argmax(dim=-1)
```

后面 `gather(next_qa_values_BA, ...)` 拿到的就成了**在线网络**的值，
「选」和「取值」塌缩成同一个网络，target 网络彻底不参与。数值上：

| 写法 | 选 a' 用 | 取值用 | 结果（target=[1,5,2], online=[9,3,4]） |
|---|---|---|---|
| vanilla | target | target | 5.0 |
| 覆盖版（错） | online | **online** | **9.0** ← 高估比 vanilla 还严重 |
| 正确 double-Q | online | **target** | **1.0** |

**不报错、能跑、只是学错。** PDF §2.5 那两个式子的下标是不同的（φ 选、φ' 取值），
下标就是全部机制。

**② `WANDB_MODE=disabled` 不生效，但 `wandb.init(mode="disabled")` 生效**

跑基准时想避开 W&B。环境变量被 `run_dqn.py:224` 的 `mode="online"` 覆盖（阶段 1 已记）。
第一次试着把 `setup_wandb` 打桩成 no-op —— **失败**：`Logger.log` 里无条件调
`wandb.log()`，没 init 过就抛 `You must call wandb.init() before wandb.log()`。
可行的写法是让桩去做一次禁用式 init：

```python
import scripts.run_dqn as m
m.setup_wandb = lambda **kw: wandb.init(mode="disabled")
```

这样 `wandb.log` 变成本地空操作，服务器上也不建 run。**之后所有冒烟/基准测试都该这么跑。**

### 遗留疑问

~~PDF 要求解释：MsPacman 的 train return 和 eval return 早期为什么差别很大？~~ ✅ 已答，见 `REPORT.md` §2.4。

一句话：**train 走 ε-greedy、eval 走纯贪心**（`utils.py:37` 不传 epsilon）。
决定性证据是 ε 衰减到 0.01 后两条线合并（step 100 万时 train 1460 / eval 1356，
eval 反被超过）—— 若差异另有来源，不会随 ε 一起消失。
已排除环境差异：`atari_dqn_config.make_env` 忽略 `eval` 参数，
`wrap_deepmind` 未接 `ClipRewardEnv`、`terminal_on_life_loss=False`。

⚠️ 反直觉的一点：**step 0 处 eval 反而更低**（60 vs 211）。
对随机初始化的网络取 argmax 是个**退化的确定性策略**，比均匀随机还差。
eval 不是天然更高，只是不交探索税。

---

## 阶段 3 — 超参敏感性（PDF §2.6）

在 LunarLander-v2 上选**一个**超参，跑另外 3 组设置（加上阶段 2 那次共 4 组）同图。
PDF 给的候选：learning rate / 网络结构 / exploration schedule / target network 更新频率。
报告里要在 caption 解释**为什么选这个超参**。

> 无现成 yaml，要自己加 3 个配置文件。这也是之后想加批量运行脚本（对标 `hw2/run_experiments.sh`）的地方。
> 因子多到跑不完时的正交表设计见 `../hw2/ABLATION.md`。
> 该文件已于 2026-09-02 续写后半篇：优化 vs 归因的分叉、随机搜索 vs 网格、
> Successive Halving 在 RL 里为什么便宜有限（rung 0 必须晚于 `learning_starts`）、
> 以及噪声底噪 σ 与 seed 数的预算公式。**§2.6 只是单因子 OFAT，用不到正交表**；
> 多因子那套是超纲内容，真要跑再翻。

### 选了哪个超参 & 为什么

> TODO

### 验证记录

> TODO

---

## 阶段 4 — SAC 数据流 + Bootstrapping（PDF §3.1–3.2）

PDF 原话：`run_sac.py` 的 TODO「应该跟你的 DQN run 脚本长得差不多，因为两者都是 off-policy」。

### 前置阅读

PDF §3.1 分了两档。**必读**（"you'll need to take a look at"）：

- [x] `src/scripts/run_sac.py` —— 主训练循环
- [x] `src/agents/sac_agent.py` —— 待实现的结构

**可能有用**（"you may also find useful"）：

- [x] `src/networks/critics.py` `StateActionCritic` —— 注意和 `DQNCritic` 的区别：
      DQN critic 是 obs → 每个动作一个 Q；SAC critic 是 (s, a) → 单个 Q
- [x] `src/configs/sac_config.py` —— 基础配置和超参清单
- [x] `experiments/sac/*.yaml` —— 各实验的配置

> ⚠️ PDF 里这两个路径是**过时的**：写的是 `src/networks/critic.py`（实际 `critics.py`，有 s）
> 和 `src/env_configs/sac_config.py`（实际 `src/configs/sac_config.py`，没有 `env_` 前缀）。

### TODO 清单

- [x] `run_sac.py:65` —— 选动作（`TODO(Section 3.1)`）✅ `agent.get_action(observation=...)`
- [x] `run_sac.py:90` —— 采样 `config["batch_size"]` 条（`TODO(Section 3.1)`）✅ 对标 `run_dqn.py:131-144`
- [x] `sac_agent.py:206` `update_critic` —— 从 actor 采样并算 next Q（`TODO(Section 3.2)`）
- [x] `sac_agent.py:227` `update_critic` —— 算 target Q（`TODO(Section 3.2)`）
- [x] `sac_agent.py:236,240` `update_critic` —— 预测 + loss（`TODO(Section 3.2)`）
- [x] `sac_agent.py:367` `update` —— 更新 critic `num_critic_updates` 次（`TODO(Section 3.2)`）
- [x] `sac_agent.py:394` `update` —— hard / soft target 更新二选一（`TODO(Section 3.2)`）

### 关键决定

**`max` 换成单样本期望，是这一节唯一的数学改动。**

```
DQN:   y = r + γ(1-d) · max_a' Q_θ̄(s', a')       枚举 A 个动作取最大，O(A)
SAC:   y = r + γ(1-d) · E_{a'~π}[Q_φ'(s', a')]    对策略取期望
                        ↑ 代码里用【一个采样】估计
```

连续动作下 A = ∞，输出层宽度不可能是无穷 —— 所以 critic 的形状必须从
`obs -> (B, A)`（`DQNCritic`，动作在**输出端**）改成 `(obs, acs) -> (B,)`
（`StateActionCritic`，动作在**输入端**，`critics.py:41` 的 `input_size = ob_dim + ac_dim`）。
形状一改，`max` 就没了着落，因为取 max 得先枚举，而"不用枚举"正是改形状的目的。
完整的动机链见 `FROM_DQN_TO_SAC.md`。

**`.sample()` 而不是 `.rsample()`**（`:207`）—— 这段在 `torch.no_grad()` 里，
目标值不能带 actor 的梯度。实测反传后 `actor` 的梯度总量为 0。
⚠️ 同一个选择在阶段 5（`entropy()`）和阶段 6（`actor_loss_reparametrize`）会再出现两次，
**那两处答案相反**，要用 `.rsample()`。

### 验证记录

> 验收命令：
> ```
> uv run src/scripts/run_sac.py -cfg experiments/sac/sanity_invertedpendulum.yaml
> ```
> 此时**还没有 actor**，return 不会高。看的是 **Q 值是否稳定在一个合理值** ——
> 发散到无穷或恒为零都说明有 bug。PDF 说这一节没有交付物，过了就往下走。

#### 这个 sanity check 的设计：把无解的问题变成有闭式解的题

「我的 SAC 写对了吗」没法直接检验 —— 没有 ground truth 可对。
这次 run 的做法是**把 actor 冻住**（`:373` 的 `actor_info = {}` 还锁着），问题就变了：

```
原问题：  我的 SAC 能学好策略吗？                      ← 无标准答案，只能"看着像"
冻住后：  critic 能正确评估【一个固定策略】π₀ 吗？      ← 策略评估，有闭式解
```

而 InvertedPendulum 的奖励结构简单到能笔算 —— **这就是作业用它而不是 HalfCheetah
做 sanity check 的原因**。

#### 闭式解

```
每步 reward = +1，杆倒即终止。某状态还剩 k 步寿命：

    Q(s,a) = 1 + γ + γ² + ... + γ^(k-1) = (1 - γ^k)/(1 - γ)

实测固定策略的存活步数（各 200 条 rollout）：
    均匀随机动作     L = 6.50  (min 3, max 22)
    未训练的 actor   L = 5.67  (min 3, max 19)

    k = 5.7（episode 开头）        ->  Q ≈ 5.5
    k ≈ 2.8（buffer 里的平均状态）  ->  Q ≈ 2.8
    => 正确答案落在【个位数】
```

#### 失败指纹：每个 bug 把 Q 推到一个可区分的确切数值

**这才是检测机制** —— 不是"看着不对劲"，而是不同错误产生互相区分得开的数：

| 若这里写错 | 数学后果 | `q_values` 会收敛到 |
|---|---|---|
| **`(1-d)` 漏了/失效** | 级数永不截断 | **1/(1-γ) = 100** |
| loss 没 backward / optimizer 没 step | 参数不动 | **≈ 0 且完全不变** |
| target 用了在线网络 `self.critic` | 靶子跟着枪跑，正反馈 | **→ ∞ 发散** |
| target 从不同步 | 靶子永远是随机网络 | 随机常数，且 target/q 的差不收敛 |
| γ 传错 | 级数和改变 | 系统性偏离（γ=0.9 → Q≈4.4） |

`100` 和 `0` 不是"偏大偏小"，是**特定错误的确切数学后果**，所以能一次性排除。

#### 正式验收 —— `exp/InvertedPendulum-v4_sac_sd1_20260902_143509`

```
   step   q_values  target_values  critic_loss
   5000      0.056          1.050       0.9907    ← 起点接近 0
   6000      3.586          3.502       0.7722    ← 1000 步内爬起来
   8000      5.201          5.051       1.8403    ← 到位
  15000      5.160          4.903       2.1198
  25000      4.903          4.953       1.7770
  35000      4.589          4.480       1.8285
  49000      4.923          4.950       2.5657
```

| 判据 | 预期 | 实测 | |
|---|---|---|---|
| 收敛到合理值 | 闭式解 2.8 ~ 5.5 | **5.061 ± 0.270**（末 20%） | ✅ |
| 不发散 | — | step≥10000 斜率 4.65e-07/step，R²=0.001，\|t\|=0.15 | ✅ |
| 不恒零 | ≠ 0 | 0.056 → 3.6（1000 步内） | ✅ |
| **`(1-d)` 生效** | ≠ 100 | 5.06，差 20 倍 | ✅ |
| target 贴住 q | — | 5.103 vs 5.061，**差 0.041**（0.8%） | ✅ 回归收敛 |

第二层判据（`target_values` vs `q_values`）验的是**回归本身收敛了没有**，
和"数值对不对"相互独立：即使 Q 碰巧对，两条持续背离就说明
`loss.backward()` → `optimizer.step()` 这条链没生效。

#### 第三层：验证前提本身（actor 真的冻住了吗）

前两层都建立在"actor 是冻的"这个前提上，前提不成立就不是在测策略评估。
`Eval_AverageReturn` 的趋势检验 |t|=1.99（n=10，卡在显著性边界），光看统计说不清，
所以直接查参数 —— 把 `agent.pt` 和**同 seed(=1) 全新初始化**逐张量比：

```
actor.net.{0,2,4}.{weight,bias}   最大差值 0.000e+00     完全相同 6/6
critics.0.net.0.weight（对照组）   最大差值 1.690e+00
critics.0.net.2.weight（对照组）   最大差值 2.733e+00
```

跑了 5 万步 actor 一个比特没动 ⇒ Eval return 那点上升（5.4→7.6，峰值 11.9）
纯粹是每个评估点只有 10 条轨迹的采样噪声（该策略的 episode 长度分布 min 3 / max 19）。

#### 这次 run 测不到什么

```
熵 bonus          :213 的 TODO 是空的 → 这次跑的是【无熵】版本
                  （yaml 里 use_entropy_bonus: true，但代码没实现，不崩也不生效）
actor 更新         :373 的 actor_info = {} 锁着
clipped double-Q  num_critic_networks=1 + backup_type=mean
                  → q_backup_strategy 的 "min" 分支（现仍是 next_qs = None）走不到，所以不崩
自动温度           这个 yaml 里没开
hard target 更新   六个 sac yaml 全是 soft
                  → :394 的 step % period 分支【整个作业都不会被执行】
```

一句话总结这个 sanity check 的设计：
**冻结一半系统，让剩下的一半退化成有闭式解的问题，再用闭式解当标尺。**

### 踩的坑

**① `=` 写成 `:`，把 `torch.distributions.Distribution` 整个覆盖掉** —— 想写类型标注，
结果写成了链式赋值：

```python
next_action_distribution = torch.distributions.Distribution = self.actor(next_obs)
#                        ↑ 等号，不是冒号 ⇒ 两个名字都被赋成那个分布对象
```

实测后果：`torch.distributions.Distribution` 从类变成了一个 `Normal` 实例，
`isinstance(x, Distribution)` 抛 `TypeError: arg 2 must be a type`。
**而且不会立刻报错**，要等别处有人做 isinstance 才炸。

**② `1 - done` 抛 RuntimeError** —— 和阶段 1 的坑①**完全一样**，隔了一个作业阶段又踩了。
`done` 是 `torch.bool`（`replay_buffer.py:50-51` 的 `np.array(bool)`），
必须 `1 - done.float()`。

**③ `nn.MSELoss(a, b)` 不是"算 MSE"，是"造一个 MSELoss 模块"** ——
`a`/`b` 被当成构造参数 `size_average`/`reduce`，报的错完全不指向真实原因：
`RuntimeError: Boolean value of Tensor with more than one value is ambiguous`。

```
nn.MSELoss        是 <class 'type'>                          ← 类
self.critic_loss  是 <class 'torch.nn.modules.loss.MSELoss'>  ← 实例，:97 已经造好了
```

**要调用的是 `:97` 那个已经造好的实例。** 阶段 1 的坑③也是围着 `self.critic_loss` 转的，
同一个对象、两种不同的误用方式。

**④ 取模的两个操作数写反** —— 写成 `self.target_update_period % step`，正确是
`step % self.target_update_period`（`dqn_agent.py:161` 自己写过一模一样的）。
后果不是"差不多"：

```
K = 1000 时，K % step == 0 的 step:  [1,2,4,5,8,10,20,25,40,50,100,125,200,250,500,1000]
  → 全挤在前 1000 步；step > 1000 之后 K % step 恒等于 K，永远不为 0
  → 【1000 步之后 target 再也不更新】
step % K == 0 的 step: [1000, 2000, 3000, 4000, 5000]  ← 正确
另外 step=0 时 K % step 直接 ZeroDivisionError
```

**⑤ 判据挑错了字段** —— 第一版把分支条件写成 `if self.auto_tune_temperature:`。
那是 §3.5 的**温度自动调节**开关，和 target 更新毫无关系。
正确判据 `:385-386` 的注释已经直接写了：看 `target_update_period` /
`soft_target_update_rate` **哪个是 `None`**，这个"恰好一个非 None"的性质由
`sac_config.py:128-133` 保证（yaml 的 `use_soft_target_update` 把其中一个掐成 `None`），
`sac_agent.py:51-53` 的 assert 兜底。**所以只判断其中一个就够，`and` 是冗余的。**

**⑥ 少写一个「每 K 步」的条件 = 等于没有 target 网络** —— hard 分支最初漏了取模判断，
变成每步硬拷贝。实测 target 与 online 的最大参数差**全程为 0.00000**：
目标 y 里的 `Q_target` 就是正在被优化的那个 `Q` 本身，靶子跟着枪动。
这不是"更新太勤"，是 target 网络机制**整个失效**。

**⑦ wandb 的 `~/.netrc` 里存了一个坏 key** —— 提示里选 (2)「Use an existing account」后
粘贴的字符串有 **168 个字符**（wandb 的 API key 是 **40 位十六进制**），
于是 wandb 认为"已配置 key"、不再提示，但认证返回
`401 {"message":"user is not logged in"}`，崩在 `run_sac.py:175` 的 `make_logger`，
**比训练循环还早**。修法：`wandb login --relogin`，只复制 https://wandb.ai/authorize
页面上那 40 个字符。
另注（延续阶段 1 的坑⑨）：`WANDB_MODE=offline` / `disabled` **都不生效**，
因为 `mode="online"` 是显式传给 `wandb.init()` 的，优先级高于环境变量。

### 遗留疑问

- **`q_values` 收敛到 5.06，比按 buffer 状态分布算的 3.5 高约 1.4 倍。**
  按均匀采样 buffer、episode 长 L≈6.5 算，平均剩余寿命 (L+1)/2 ≈ 3.75 ⇒ Q ≈ 3.7。
  实测偏高。可能是函数逼近误差、也可能是 buffer 里前 1000 步纯随机段的分布偏移。
  没深究 —— 闭式解本来就只用来定**量级**，不是精确对表。若阶段 8 讨论高估偏差时
  需要一个基线，回来把这个差距算清楚。
- **hard target 更新分支零覆盖。** 六个 sac yaml 全是 soft，`:394` 的
  `step % period` 那一支只有单元测试跑过，**没有任何一次作业 run 会执行它**。
  那里若有 bug，作业照样交得出去，但代码是错的。
- `exp/` 下留了两个空目录（`..._20260902_142642`、`..._20260902_143152`），
  是 wandb 崩在 `make_logger` 时 `os.makedirs` 已经建好、训练还没开始造成的。
  **打包 `submit.zip` 前要删。**
- `run_sac.py:92-98` 和 `sac_agent.py:200-203` 各留了一个三引号草稿块。
  不是 docstring（不在函数体首行），是裸字符串表达式，CPython 会当常量丢掉、零运行开销，
  但会跟着 `submit.zip` 交上去。按 NOTES 开头定的分界线该删。

---

## 阶段 5 — 熵 bonus（PDF §3.3）

### TODO 清单

- [ ] `sac_agent.py` `entropy()` —— 近似熵（`TODO(Section 3.3)`）
- [ ] `sac_agent.py` `update_critic` —— target 值里加熵项（`TODO(Section 3.3)`）
- [ ] `sac_agent.py` `update_actor` —— actor loss 里加熵项（`TODO(Section 3.3)`）
- [ ] `sac_agent.py` `update` —— 启用 `self.update_actor()`（`TODO(Section 3.3)`）

### 验证记录

> 验收：同样跑 `sanity_invertedpendulum.yaml`。此时 actor loss **只有熵项**（还没做重参数化，
> 没有任何最大化奖励的东西），所以 entropy 应该一路涨到接近动作空间的最大熵。
> 1 维 tanh-squashed 动作空间 ⇒ **≈ log 2 ≈ 0.69**。显著高于或低于都说明有 bug。

> TODO

### 踩的坑

> TODO

---

## 阶段 6 — 重参数化 actor（PDF §3.4）

### TODO 清单

- [ ] `sac_agent.py` `actor_loss_reparametrize` —— 采样（`TODO(Section 3.4)`，PDF 提示用 `.rsample()`）
- [ ] `sac_agent.py` `actor_loss_reparametrize` —— 算 Q（`TODO(Section 3.4)`）
- [ ] `sac_agent.py` `actor_loss_reparametrize` —— 算 loss（`TODO(Section 3.4)`）

### 验证记录

> 验收命令：
> ```
> uv run src/scripts/run_sac.py -cfg experiments/sac/sanity_invertedpendulum.yaml   # 期望 ≈ 1000
> uv run src/scripts/run_sac.py -cfg experiments/sac/halfcheetah.yaml               # 期望 ≥ 6000，约 3h
> ```

> TODO

### 踩的坑

> TODO

---

## 阶段 7 — 自动温度（PDF §3.5）

把温度 β 的选择写成带熵约束的对偶问题，用对偶梯度下降在线调 α。原论文：arXiv 1812.05905 §5。

### TODO 清单

- [ ] `sac_agent.py` `__init__` —— `self.log_alpha`（可学习参数）、`self.alpha_optimizer`、
      `self.target_entropy = -action_dim`（`TODO(Section 3.5)`）
- [ ] `sac_agent.py` `get_temperature()` —— 自动调温时返回学到的温度（`TODO(Section 3.5)`）
- [ ] `sac_agent.py` `update_alpha()` —— 对偶梯度（`TODO(Section 3.5)`）
- [ ] `sac_agent.py` `update()` —— 调用 `update_alpha()`

> 实现要点（PDF 明写）：用 α = exp(log α) 保证 α > 0；log α 是可学习参数且有**独立的** optimizer；
> 每次 actor/critic 更新之后再更新 log α。

### 验证记录

> 验收命令：
> ```
> uv run src/scripts/run_sac.py -cfg experiments/sac/sanity_invertedpendulum_autotune.yaml  # 仍需 ≈ 1000
> uv run src/scripts/run_sac.py -cfg experiments/sac/halfcheetah_autotune.yaml
> ```

> TODO

### 遗留疑问

> PDF 要求回答三问：
> 1. 自动调温相比固定温度是变好了还是打平？
> 2. 训练中温度怎么演化 —— 升、降、还是稳住？
> 3. 为什么在 HalfCheetah 上会是这个走向？
>
> PDF 提醒：默认固定温度对 HalfCheetah 本来就调得不错，自动调温**不一定更好**；
> 它的价值在于换个新环境时不用手调。
>
> Optional：扫 β ∈ {0.01, 0.05, 0.1, 0.5, 1.0} 和 autotune 同图，看性能对固定温度有多敏感。

---

## 阶段 8 — clipped double-Q（PDF §3.6）

### TODO 清单

- [ ] `sac_agent.py` `q_backup_strategy` 的 `"min"` 分支（`TODO(Section 3.6)`）
      —— `"mean"`（单 Q）分支已经给好了

### 验证记录

> 验收命令：
> ```
> uv run src/scripts/run_sac.py -cfg experiments/sac/hopper_singleq.yaml
> uv run src/scripts/run_sac.py -cfg experiments/sac/hopper_clipq.yaml     # 期望 ≥ 1500
> ```
> 两者的 **eval return 和 q_values 都要画**，并讨论与高估偏差的关系。

> TODO

### 遗留疑问

> TODO

---

## 提交物清单（PDF §4）

打包命令：`zip -r submit.zip exp src README.md pyproject.toml uv.lock`（< 100MB，`exp` 和 `src` **不要**再套一层父目录）。

`exp/` 里必须有这 7 个 run（按前缀识别，即目录名 `_sd` 之前的部分）：

- [x] `CartPole-v1_dqn_sd*` ✅ `exp/CartPole-v1_dqn_sd1_20260825_151358/`
- [x] `LunarLander-v2_dqn_sd*` ✅ `exp/LunarLander-v2_dqn_sd1_20260825_153744/`
- [x] `MsPacman_dqn_sd*` ✅ `exp/MsPacman_dqn_sd1_20260825_155618/`
- [ ] `HalfCheetah-v4_sac_sd*`
- [ ] `HalfCheetah-v4_sac_autotune_sd*`
- [ ] `Hopper-v4_sac_singleq_sd*`
- [ ] `Hopper-v4_sac_clipq_sd*`

> §2.6 的超参 sweep 不在这个必需列表里（它们是 LunarLander 的额外 run），但图要进报告。
> PDF 说本次只要求交**最好的一次 run**，不要求多 seed 平均。

---

## 跨作业复用

> 留到 hw3 全部做完再回填 —— 挑出对 hw4 / hw5 / final project 有用的结论，
> 以及该往 `../hw2/SHAPES.md`、`../hw2/FORMULA_TO_CODE.md` 里续写的条目。

> TODO
