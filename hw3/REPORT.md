# HW3 报告：Q-Learning 与 Actor-Critic

> **交付去向**
> - 本文件导出 PDF → Gradescope 的 **HW3 Report**
> - 代码 + `exp/` 日志 → **HW3 Code**（submit.zip，<100MB）
> - 图片放 `hw3/report/`，由 `uv run plot_results.py <阶段>` 生成
>
> **填写方式**：每节的命令表跑完一行就填一行。命令不用手抄，
> 每个 run 的 `exp/<run目录>/flags.json` 里有完整参数，直接复制。
>
> **导出 PDF**（在 `hw3/` 目录下执行，图片用的是相对路径，必须从这里跑）：
>
> ```bash
> pandoc REPORT.md -o REPORT.pdf --pdf-engine=xelatex \
>   -V mainfont="DejaVu Sans" -V monofont="DejaVu Sans Mono" \
>   -V CJKmainfont="Noto Sans CJK SC" \
>   -V geometry:margin=2cm -V colorlinks=true
> ```
>
> 三个字体缺一不可（沿用 hw2 已验证的组合）：`CJKmainfont` 管中文，
> `mainfont` 管表格里的 ✓/✗，`monofont` 管代码块里的希腊字母（γ ε φ）。
> 少一个对应字符会**静默变成空白**，导出后务必翻一遍 PDF。

---

## 实验 1 — CartPole-v1 基础 DQN（PDF §2.4）

### 1.1 运行记录

| run 目录 | seed | 设备 | 总步数 | 评估点 | 达 500 次数 | 首次达标 |
|---|---|---|---|---|---|---|
| `CartPole-v1_dqn_sd1_20260825_151358` | 1 | CPU（`--no_gpu`） | 100,000 | 40 | **10** | step **10,000** |

配置 `experiments/dqn/cartpole.yaml`：`hidden_size=64`、`num_layers=2`、`lr=5e-4`、
`discount=0.99`、`target_update_period=1000`、`batch_size=128`、`learning_starts=1000`、
`use_double_q=false`。

### 1.2 图

![CartPole-v1 基础 DQN 的评估回报随环境步数变化；每 2500 步用 10 条轨迹评估一次，500 为该环境上限](report/stage1_cartpole.png)

**图未做平滑。** PDF 的验收判据是「训练中至少一次达到 500」——这是尖峰事件不是趋势，
滑动平均会把峰削掉，让图读起来像没达标。40 个评估点也足够稀疏，直接画原始值即可。

曲线中段（15,000–50,000）回落到 80~200 是 DQN 的常态，不是实现问题：ε 那时还在 0.4 以上、
replay buffer 里旧策略的数据占比高、target 网络每 1000 步才同步一次。
ε 衰减到 0.1 附近（step 30,000 之后）曲线才稳定在高位。

### 1.3 完整命令

```bash
uv run src/scripts/run_dqn.py -cfg experiments/dqn/cartpole.yaml --eval_interval 2500 --no_gpu
```

> `--no_gpu` 不是笔误：CartPole 的 critic 只有 4,610 个参数，实测 CPU **1038 it/s** vs
> GPU **871 it/s**（20,000 步基准），CPU 快 19%。瓶颈在 `env.step()` 和逐步 `get_action`
> 的 host→device 拷贝，不在矩阵乘法。LunarLander 网络大 15 倍后这个结论会翻转，见 §2.3。

---

## 实验 2 — LunarLander-v2 Double-Q（PDF §2.5）

### 2.1 运行记录

| run 目录 | seed | 设备 | 总步数 | 评估点 | 达 200 次数 | 首次达标 | 最好 |
|---|---|---|---|---|---|---|---|
| `LunarLander-v2_dqn_sd1_20260825_153744` | 1 | GPU (RTX 4090) | 500,000 | 50 | **17** | step **260,000** | **274.0** |

配置 `experiments/dqn/lunarlander.yaml`：`hidden_size=256`、`num_layers=2`、`lr=1e-3`、
`discount=0.99`、`target_update_period=1000`、`batch_size=64`、`learning_starts=20000`、
**`use_double_q=true`**。最终评估点 224.7。

达标不是单点偶然：step 260,000 首次越线后，300,000 之后的 20 个评估点里有 14 个 ≥200。

### 2.2 图

![LunarLander-v2 Double-Q DQN 的评估回报随环境步数变化；每 10000 步用 10 条轨迹评估一次](report/stage2_lunarlander.png)

### 2.3 完整命令

```bash
uv run src/scripts/run_dqn.py -cfg experiments/dqn/lunarlander.yaml
```

> 这里**不加** `--no_gpu`。LunarLander 的 critic 是 8→256→256→4 共 69,124 个参数，
> 是 CartPole 的 15 倍，矩阵乘终于压过传输开销：实测 GPU **493 it/s** vs CPU **369 it/s**，
> GPU 快 34%。

### 2.4 MsPacman

> 待跑。PDF 要求把 `train return` 和 `eval return` 画在同一张图上并解释早期差异。

---

## 实验 3 — 超参敏感性（PDF §2.6）

> 待做。在 LunarLander 上选一个超参跑另外 3 组设置，四条曲线同图，
> caption 里要解释**为什么选这个超参**。

---

## 实验 4 — SAC（PDF §3）

> 待做。§3.4 HalfCheetah ≥6000、§3.5 自动温度对比图 + 三问、§3.6 Hopper 单 Q vs clipped double-Q。
