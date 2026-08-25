"""从 exp/ 的 log.csv 画 REPORT.md 需要的学习曲线。

用法：  uv run plot_results.py exp1

PDF §3.2 的要求：
  - 两张图：小 batch（前缀 cartpole 不含 lb）/ 大 batch（前缀 cartpole_lb）
  - y 轴 average return，x 轴 **Train_EnvstepsSoFar**（不是迭代数）
"""
import csv, glob, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# dataviz 参考调色板（light 模式 categorical slot 1-4，已过验证器）
# dataviz 参考调色板 light 模式 categorical slot 1-5，五色一起过验证器：
#   CVD 最差相邻对 ΔE 9.1、常规视觉最差 19.6，均达标。
#   aqua/yellow/magenta 对比度 < 3:1，靠 legend + REPORT 数据表兜底（relief 规则）。
SERIES  = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; GRID = "#e4e3df"

for cand in ("Noto Sans CJK JP", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"):
    if any(cand in f.name for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand; break
plt.rcParams["axes.unicode_minus"] = False


def load(run_dir, ycol="Eval_AverageReturn"):
    """读一个 run 的 (x, y)。ycol 不存在时返回 None —— 例如 use_baseline=False 的 run
    没有 Baseline Loss 这一列（self.critic is None，step 4 整段跳过）。"""
    with open(os.path.join(run_dir, "log.csv")) as f:
        rows = list(csv.DictReader(f))
    if not rows or ycol not in rows[0] or rows[0][ycol] == "":
        return None
    return ([float(r["Train_EnvstepsSoFar"]) for r in rows],
            [float(r[ycol]) for r in rows])


def find(exp_name):
    """按 exp_name 精确定位 run 目录（目录名形如 <env>_<exp_name>_sd<seed>_<时间戳>）。"""
    hits = [d for d in glob.glob("exp/*") if os.path.basename(d).split("_sd")[0].endswith("_" + exp_name)]
    if len(hits) != 1:
        raise SystemExit(f"{exp_name}: 期望 1 个 run 目录，找到 {len(hits)} 个 -> {hits}")
    return hits[0]


def smooth(y, w):
    """滑动平均。窗口内不足 w 个点时用已有的点平均（前段不丢数据）。"""
    out = []
    for i in range(len(y)):
        lo = max(0, i - w + 1)
        out.append(sum(y[lo:i+1]) / (i - lo + 1))
    return out


def plot(names, labels, title, subtitle, outfile, ycol="Eval_AverageReturn",
         ylabel=None, logy=False, smooth_w=0):
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=200)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)

    ends = []
    for i, name in enumerate(names):
        got = load(find(name), ycol)
        if got is None:                     # 该 run 没有这一列，跳过（并说明，不静默丢）
            print(f"     跳过 {name}：log.csv 里没有 '{ycol}' 列")
            continue
        x, y = got
        if smooth_w:
            # 原始曲线淡显（不丢信息），平滑曲线实显（读趋势）。RL 学习曲线的标准画法。
            ax.plot(x, y, color=SERIES[i], lw=1, alpha=0.22, zorder=2)
            y = smooth(y, smooth_w)
        ax.plot(x, y, color=SERIES[i], lw=2, solid_joinstyle="round",
                solid_capstyle="round", label=labels[i], zorder=3)
        ax.plot(x[-1], y[-1], "o", ms=8, color=SERIES[i], mec=SURFACE, mew=2, zorder=4)
        ends.append((x[-1], y[-1]))

    # 末端直接标注在这里没有信息量：四条曲线终点都在 200 附近，标注会互相重叠且指向同一点。
    # 身份识别交给 legend（≥2 series 必须有）+ REPORT.md §1.1 的数据表（table view），
    # 这也满足 dataviz 对低对比度色槽（aqua / yellow）的 relief 要求。
    xmax = max(e[0] for e in ends)
    ax.set_xlim(0, xmax * 1.02)

    ax.set_title(title, fontsize=12, color=INK, loc="left", pad=18, fontweight="bold")
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")
    ax.set_xlabel("Train_EnvstepsSoFar（环境步数）", fontsize=9, color=INK2)
    ax.set_ylabel(ylabel or ycol, fontsize=9, color=INK2)
    if logy:
        ax.set_yscale("log")
    ax.grid(axis="y", color=GRID, lw=1, ls="-"); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right", ncol=2)
    fig.tight_layout()
    os.makedirs("report", exist_ok=True)
    fig.savefig(os.path.join("report", outfile), facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig); print(f"  OK  report/{outfile}")


def plot_l9_grid(outfile="exp4_l9_grid.png"):
    """L9 九组画成 3x3 small multiples —— 9 条曲线叠一张读不了。

    每格标注：首次到 1000 的步数、达标后 ≥950 的占比。
    """
    L9 = [(1,1,1,1),(1,2,2,2),(1,3,3,3),(2,1,2,3),(2,2,3,1),(2,3,1,2),(3,1,3,2),(3,2,1,3),(3,3,2,1)]
    LV = [{1:"γ=0.95",2:"γ=0.99",3:"γ=1.0"}, {1:"2/32",2:"2/64",3:"3/128"},
          {1:"b500",2:"b1000",3:"b2000"}, {1:"lr5e-3",2:"lr1e-2",3:"lr2e-2"}]
    fig, axes = plt.subplots(3, 3, figsize=(11, 8), dpi=170, sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for k, (cfg, ax) in enumerate(zip(L9, axes.flat), 1):
        ax.set_facecolor(SURFACE)
        got = load(find(f"pendulum_L9_{k}"))
        x, y = got
        i = next((j for j, v in enumerate(y) if v >= 1000), None)
        hit = f"{int(x[i]):,} 步" if i is not None else "未达到"
        hold = (np.array(y[i:]) >= 950).mean() if i is not None else 0.0
        col = SERIES[0] if i is not None and hold >= 0.65 else (SERIES[3] if i is not None else SERIES[1])
        ax.plot(x, y, color=col, lw=1, alpha=0.3, zorder=2)
        ax.plot(x, smooth(y, 8), color=col, lw=1.8, zorder=3)
        if i is not None:
            ax.axvline(x[i], color=GRID, lw=1, zorder=1)
        ax.axhline(1000, color=GRID, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.set_title(f"#{k}  " + " ".join(LV[c][cfg[c]] for c in range(4)),
                     fontsize=8, color=INK, loc="left", pad=3)
        ax.text(0.98, 0.06, f"首次 {hit}\n达标后≥950 {hold:.0%}", transform=ax.transAxes,
                fontsize=7, color=INK2, ha="right", va="bottom")
        ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=7, length=0)
    fig.suptitle("InvertedPendulum-v4  L9 正交表九组学习曲线", fontsize=13,
                 color=INK, x=0.01, ha="left", fontweight="bold")
    fig.text(0.01, 0.945, "蓝=达标且稳（达标后≥950 占比 ≥65%）  黄=达标但不稳  橙=未达标；"
             "竖线为首次到 1000，虚线为满分 1000", fontsize=8, color=INK2, ha="left")
    fig.supxlabel("Train_EnvstepsSoFar（环境步数）", fontsize=9, color=INK2)
    fig.supylabel("Eval_AverageReturn", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0.01, 0.01, 1, 0.93))
    os.makedirs("report", exist_ok=True)
    fig.savefig(os.path.join("report", outfile), facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig); print(f"  OK  report/{outfile}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "exp1"
    if which == "exp1":
        L = ["无 rtg / 无 na", "-rtg", "-na", "-rtg -na"]
        plot(["cartpole", "cartpole_rtg", "cartpole_na", "cartpole_rtg_na"], L,
             "CartPole-v0 学习曲线 —— 小 batch (b=1000)",
             "四种 return estimator / advantage 归一化组合；200 为该环境的 return 上限", "exp1_small_batch.png")
        plot(["cartpole_lb", "cartpole_lb_rtg", "cartpole_lb_na", "cartpole_lb_rtg_na"], L,
             "CartPole-v0 学习曲线 —— 大 batch (b=4000)",
             "四种 return estimator / advantage 归一化组合；200 为该环境的 return 上限", "exp1_large_batch.png")
    elif which == "exp2":
        BASE = [
            "cheetah_baseline",
            "cheetah_baseline_blr0.001",
            "cheetah_baseline_bgs1",
        ]
        BL = [
            "默认：-blr 0.01 / -bgs 5",
            "仅降 -blr：0.001",
            "仅降 -bgs：1",
        ]
        # 图一：baseline loss —— cheetah 没有 baseline，不参与
        plot(BASE, BL,
             "HalfCheetah-v4 baseline loss —— critic 的回归误差",
             "分别降低 critic 学习率与每轮更新次数；纵轴对数刻度",
             "exp2_baseline_loss.png", ycol="Baseline Loss",
             ylabel="Baseline Loss (MSE)", logy=True)
        # 图二：eval return —— 无 baseline + 默认 baseline + 两个单变量消融
        plot(["cheetah"] + BASE, ["无 baseline"] + BL,
             "HalfCheetah-v4 eval return",
             "验收线：cheetah_baseline 末尾 > 300",
             "exp2_eval_return.png")
        # Optional 直接对比：突出 advantage normalization 对 actor 学习的影响。
        # na run 同时开启了视频采样，因此图中明确标注，不将差异全部归因于 -na。
        plot(
            ["cheetah_baseline", "cheetah_baseline_na_video"],
            ["默认 baseline", "baseline + -na (+ video)"],
            "HalfCheetah-v4 advantage normalization 对比",
            "两者 critic 超参相同；optional run 另开启了视频采样",
            "exp2_na_comparison.png",
        )
    elif which == "exp3":
        # PDF §5 明确要求 "a single plot" —— 五条 λ 曲线画在同一张
        LAMS = ["0", "0.95", "0.98", "0.99", "1"]
        plot([f"lunar_lander_lambda{l}" for l in LAMS],
             [f"λ={l}" for l in LAMS],
             "LunarLander-v2 GAE λ 扫描",
             "λ=0 退化成单步 TD（式 16），λ=1 退化成纯 Monte Carlo；粗线为 10 点滑动平均，淡线为原始值",
             "exp3_lambda_sweep.png", smooth_w=10)
    elif which == "exp4-l9":
        plot_l9_grid()
    else:
        raise SystemExit(f"未知实验：{which}")
