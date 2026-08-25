"""从 exp/ 的 log.csv 画 REPORT.md 需要的学习曲线。

用法：  uv run plot_results.py stage1     # CartPole（PDF §2.4 交付物）
        uv run plot_results.py stage2     # LunarLander（PDF §2.5 交付物）

移植自 hw2/plot_results.py。两处关键差异：
  - hw3 的横轴列名是 `step`，不是 `Train_EnvstepsSoFar`
  - hw3 的 log.csv 里评估行和训练行交错，评估行之外 Eval_AverageReturn 为空，必须先过滤
"""
import csv, glob, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# dataviz 参考调色板 light 模式 categorical slot 1（已过 validate_palette.js：
# 亮度带 / 彩度下限 / 对比度 ≥3:1 全 PASS）。单序列图不需要 legend —— 标题即身份。
SERIES  = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; GRID = "#e4e3df"

for cand in ("Noto Sans CJK JP", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"):
    if any(cand in f.name for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand; break
plt.rcParams["axes.unicode_minus"] = False


def load(run_dir, ycol="Eval_AverageReturn"):
    """读一个 run 的评估曲线 (x, y)。训练行的 ycol 是空串，跳过。"""
    with open(os.path.join(run_dir, "log.csv")) as f:
        rows = [r for r in csv.DictReader(f) if r.get(ycol)]
    if not rows:
        return None
    return ([float(r["step"]) for r in rows], [float(r[ycol]) for r in rows])


def find(prefix):
    """按 `<env>_<exp_name>` 前缀定位 run 目录。

    hw2 那版按 exp_name 后缀匹配，hw3 不能照抄 —— CartPole 和 LunarLander 的
    exp_name 都是 `dqn`，只按它匹配会同时命中两个。
    """
    hits = [d for d in glob.glob("exp/*") if os.path.basename(d).startswith(prefix + "_sd")]
    if len(hits) != 1:
        raise SystemExit(f"{prefix}: 期望 1 个 run 目录，找到 {len(hits)} 个 -> {hits}")
    return hits[0]


def smooth(y, w):
    """滑动平均。窗口不足 w 个点时用已有的点平均（前段不丢数据）。"""
    return [sum(y[max(0, i-w+1):i+1]) / (i - max(0, i-w+1) + 1) for i in range(len(y))]


def plot(prefix, title, subtitle, outfile, threshold):
    """画原始评估曲线。

    **不做平滑**。hw2 那版把原始值淡显、平滑值实显，适合几百个噪声点的曲线；
    这里只有 40~50 个评估点，而且 PDF 的验收判据是「至少一次达到」——
    是个尖峰事件，不是趋势。平滑会把峰削掉，让图读起来像没达标，
    与要证明的结论正好相反。
    """
    x, y = load(find(prefix))
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=200)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)

    # 达标线用中性墨色，不占系列色槽 —— 它是参考规则，不是一条数据序列。
    # 标签放左上，避开右侧的终点圆点。
    ax.axhline(threshold, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(x[-1] * 0.012, threshold, f"验收线 {threshold:g}", fontsize=8, color=INK2,
            va="bottom", ha="left")

    ax.plot(x, y, color=SERIES[0], lw=2, zorder=3,
            solid_joinstyle="round", solid_capstyle="round")
    ax.plot(x[-1], y[-1], "o", ms=8, color=SERIES[0], mec=SURFACE, mew=2, zorder=4)

    # 只标首次达标这一个点 —— 它就是 PDF 的验收判据，别把每个点都标上。
    hits = [j for j, v in enumerate(y) if v >= threshold]
    if hits:
        i = hits[0]
        ax.plot(x[i], y[i], "o", ms=8, color=SERIES[0], mec=SURFACE, mew=2, zorder=4)
        # 偏移量要够大：首次达标之后曲线常常仍在同一高度延伸（LunarLander 就是），
        # 小偏移会让文字和引线压在数据上。往下拉到空白区。
        ax.annotate(f"首次达标  step {int(x[i]):,}（共 {len(hits)} 次）",
                    xy=(x[i], y[i]), xytext=(16, -78), textcoords="offset points",
                    fontsize=8, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8,
                                    shrinkA=0, shrinkB=4))

    ax.set_xlim(0, x[-1] * 1.02)
    ax.set_ylim(min(0, min(y) * 1.05), max(y) * 1.14)
    ax.set_title(title, fontsize=12, color=INK, loc="left", pad=18, fontweight="bold")
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")
    ax.set_xlabel("环境步数（step）", fontsize=9, color=INK2)
    ax.set_ylabel("Eval_AverageReturn", fontsize=9, color=INK2)
    ax.grid(axis="y", color=GRID, lw=1, ls="-"); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    fig.tight_layout()
    os.makedirs("report", exist_ok=True)
    fig.savefig(os.path.join("report", outfile), facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig); print(f"  OK  report/{outfile}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "stage1"
    if which == "stage1":
        plot("CartPole-v1_dqn",
             "CartPole-v1 —— 基础 DQN 学习曲线（PDF §2.4）",
             "每 2500 步评估 10 条轨迹取平均；500 是该环境的 return 上限",
             "stage1_cartpole.png", threshold=500)
    elif which == "stage2":
        plot("LunarLander-v2_dqn",
             "LunarLander-v2 —— Double-Q DQN 学习曲线（PDF §2.5）",
             "每 10000 步评估 10 条轨迹取平均；use_double_q=true",
             "stage2_lunarlander.png", threshold=200)
    else:
        raise SystemExit(f"未知阶段：{which}")
