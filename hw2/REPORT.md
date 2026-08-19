# HW2 报告：Policy Gradients

> **交付去向**
> - 本文件导出 PDF → Gradescope 的 **HW2 Report**
> - 代码 + `exp/` 日志 → **HW2 Code**（submit.zip，<100MB）
> - 图片放 `hw2/report/`，命名见各节
>
> **填写方式**：每节的命令表跑完一行就填一行。命令不用手抄，
> 每个 run 的 `exp/<run目录>/flags.json` 里有完整参数，直接复制。

---

## 实验 1 — CartPole-v0（PDF §3.2）

### 1.1 运行记录

| exp_name | -b | -rtg | -na | 最终 Eval_AverageReturn | 是否收敛到 200 | run 目录 |
|---|---|---|---|---|---|---|
| `cartpole`          | 1000 | ✗ | ✗ | | | |
| `cartpole_rtg`      | 1000 | ✓ | ✗ | | | |
| `cartpole_na`       | 1000 | ✗ | ✓ | | | |
| `cartpole_rtg_na`   | 1000 | ✓ | ✓ | | | |
| `cartpole_lb`       | 4000 | ✗ | ✗ | | | |
| `cartpole_lb_rtg`   | 4000 | ✓ | ✗ | | | |
| `cartpole_lb_na`    | 4000 | ✗ | ✓ | | | |
| `cartpole_lb_rtg_na`| 4000 | ✓ | ✓ | | | |

### 1.2 图

x 轴一律用 `Train_EnvstepsSoFar`，**不是**迭代数。

- 小 batch 对比（4 条 `cartpole` 前缀曲线）：`report/exp1_small_batch.png`
- 大 batch 对比（4 条 `cartpole_lb` 前缀曲线）：`report/exp1_large_batch.png`

### 1.3 问题

**Q1. 不做 advantage normalization 时，trajectory-centric 和 reward-to-go 哪个表现更好？**

> TODO

**Q2. 两种 value estimator，为什么其中一个通常更受偏好？**

> TODO（提示：往方差上想，PDF §2.2.1 的 causality 论证）

**Q3. Advantage normalization 有帮助吗？**

> TODO

**Q4. Batch size 有影响吗？**

> TODO

### 1.4 完整命令

```bash
# TODO: 从 flags.json 回填，含任何偏离默认值的参数
```

---

## 实验 2 — HalfCheetah-v4 baseline（PDF §4.2）

### 2.1 运行记录

| exp_name | use_baseline | -blr | -bgs | 最终 Eval_AverageReturn | run 目录 |
|---|---|---|---|---|---|
| `cheetah`          | ✗ | — | — | | |
| `cheetah_baseline` | ✓ | 0.01 | 5 | | |
| `cheetah_baseline_<变体>` | ✓ | | | | |

**验收线**：`cheetah_baseline` 末尾 Eval return 要 > 300。跑 2–3 次还到不了就回去查实现。

### 2.2 图

- baseline loss 曲线：`report/exp2_baseline_loss.png`
- eval return 曲线：`report/exp2_eval_return.png`

### 2.3 问题

**Q. 降低 `-bgs` 或 `-blr` 后，(a) baseline 学习曲线怎么变？(b) 策略性能怎么变？**

> TODO（记清楚你改的是哪一个、从多少改到多少）

### 2.4 完整命令

```bash
# TODO
```

### 2.5 可选

- [ ] 加回 `-na` 看提升多少
- [ ] `--video_log_freq 10`，在 WandB 里看 HalfCheetah 跑起来

---

## 实验 3 — LunarLander-v2 GAE（PDF §5）

其余超参一律不许动，只扫 λ。

### 3.1 运行记录

| λ | 训练中出现过的最高 Eval_AverageReturn | 最终 Eval_AverageReturn | run 目录 |
|---|---|---|---|
| 0    | | | |
| 0.95 | | | |
| 0.98 | | | |
| 0.99 | | | |
| 1    | | | |

**验收线**：最好的那次训练过程中至少出现一次 > 150。

### 3.2 图

五条 λ 曲线画在同一张：`report/exp3_lambda_sweep.png`

### 3.3 问题

**Q1. λ 如何影响任务表现？**

> TODO

**Q2. λ=0 对应什么？λ=1 对应什么？结合 LunarLander 的结果用一两句话说明。**

> TODO（提示：把 λ 代进 PDF 式 (21) 的递推里，看退化成什么 —— 一个是式 (16) 的单步 TD，一个是纯 Monte Carlo）

### 3.4 完整命令

```bash
# TODO
```

---

## 实验 4 — InvertedPendulum-v4 调参（PDF §6）

目标：**100K 环境步以内**摸到 return 1000。基线设置要 500K 步（`-n 100 -b 5000`）。

### 4.1 调参过程

> 这张表是 Q2 的原始素材 —— `flags.json` 只存了「每组是什么」，不存「你为什么这么改」。
> 每跑一组就补一行，**"改动理由" 这列不要留空**。

| # | 相对上一组改了什么 | 改动理由 | 首次到 1000 的步数 | 观察 |
|---|---|---|---|---|
| 0 | （默认基线） | — | | |
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |

### 4.2 最佳配置

```bash
# TODO: exp_name 必须以 pendulum 开头
```

| 超参 | 默认 | 我的值 |
|---|---|---|
| `--discount` | 1.0 | |
| `-n` / `-b` | 100 / 5000 | |
| `-lr` | 5e-3 | |
| `-l` / `-s` | 2 / 64 | |
| `-rtg` | ✗ | |
| `-na` | ✗ | |
| `--use_baseline` | ✗ | |
| `--gae_lambda` | None | |

### 4.3 问题

**Q. 哪些超参在调参过程中真正起了作用？**

> TODO（直接从 4.1 的表里读结论）

### 4.4 图

我的配置 vs 默认配置，x 轴环境步数：`report/exp4_pendulum.png`

---

## 提交前检查（PDF §7）

`exp/` 下必须有这 15 个 run，目录名前缀要精确匹配：

- [ ] `CartPole-v0_cartpole_sd*`
- [ ] `CartPole-v0_cartpole_rtg_sd*`
- [ ] `CartPole-v0_cartpole_na_sd*`
- [ ] `CartPole-v0_cartpole_rtg_na_sd*`
- [ ] `CartPole-v0_cartpole_lb_sd*`
- [ ] `CartPole-v0_cartpole_lb_rtg_sd*`
- [ ] `CartPole-v0_cartpole_lb_na_sd*`
- [ ] `CartPole-v0_cartpole_lb_rtg_na_sd*`
- [ ] `HalfCheetah-v4_cheetah_sd*`
- [ ] `HalfCheetah-v4_cheetah_baseline_sd*`
- [ ] `LunarLander-v2_lunar_lander_lambda0_sd*`
- [ ] `LunarLander-v2_lunar_lander_lambda0.95_sd*`
- [ ] `LunarLander-v2_lunar_lander_lambda0.98_sd*`
- [ ] `LunarLander-v2_lunar_lander_lambda0.99_sd*`
- [ ] `LunarLander-v2_lunar_lander_lambda1_sd*`
- [ ] `InvertedPendulum-v4_pendulum*_sd*`

其他：

- [ ] 每个 run 目录里有 `agent.pt` / `flags.json` / `log.csv` / `log.pkl`
- [ ] `src/` 目录结构与原始仓库一致
- [ ] 带上 `pyproject.toml` / `uv.lock` / `README.md`
- [ ] `exp` 和 `src` **平铺在 zip 根目录**，不要再套一层父目录
- [ ] submit.zip < 100MB
