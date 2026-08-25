"""L9 正交表的主效应分析。

用法：  uv run analyze_l9.py

指标：首次达到 return 1000 的环境步数（未达到记为预算上限）。
主效应 = 该水平出现的 3 组的平均指标 —— 因为正交，其他因子的影响被平均掉了。
"""
import csv, glob, os

BUDGET = 150_000
L9 = [(1,1,1,1),(1,2,2,2),(1,3,3,3),(2,1,2,3),(2,2,3,1),(2,3,1,2),(3,1,3,2),(3,2,1,3),(3,3,2,1)]
LEVELS = {
    "discount": {1: "0.95", 2: "0.99", 3: "1.0"},
    "网络 -l/-s": {1: "2/32", 2: "2/64", 3: "3/128"},
    "batch -b": {1: "500", 2: "1000", 3: "2000"},
    "lr": {1: "5e-3", 2: "1e-2", 3: "2e-2"},
}


def metric(exp_name, target=1000.0):
    """返回 (首次达标步数 或 None, 训练中最高 return)。"""
    hits = [d for d in glob.glob("exp/InvertedPendulum*")
            if os.path.basename(d).split("_sd")[0].endswith("_" + exp_name)]
    if len(hits) != 1:
        return None, None
    rows = list(csv.DictReader(open(os.path.join(hits[0], "log.csv"))))
    best = 0.0
    for r in rows:
        ev = float(r["Eval_AverageReturn"]); best = max(best, ev)
        if ev >= target:
            return int(float(r["Train_EnvstepsSoFar"])), best
    return None, best


if __name__ == "__main__":
    res = []
    print(f"{'#':>2}{'discount':>10}{'-l/-s':>9}{'-b':>7}{'-lr':>7}{'首次到1000':>12}{'最高':>9}")
    for k, row in enumerate(L9, 1):
        step, best = metric(f"pendulum_L9_{k}")
        if best is None:
            print(f"{k:>2}   (未跑或目录数不为 1)"); res.append(None); continue
        res.append((row, step if step else BUDGET, best))
        lv = [LEVELS[n][v] for n, v in zip(LEVELS, row)]
        print(f"{k:>2}{lv[0]:>10}{lv[1]:>9}{lv[2]:>7}{lv[3]:>7}"
              f"{(f'{step:,}' if step else '未达到'):>12}{best:>9.0f}")

    ok = [r for r in res if r]
    if len(ok) < 9:
        print(f"\n只有 {len(ok)}/9 组有数据，主效应先不算。"); raise SystemExit
    print("\n主效应（同一水平那 3 组的平均首次达标步数，越小越好；未达到按预算上限计）\n")
    for col, (name, lv) in enumerate(LEVELS.items()):
        cells = []
        for v in (1, 2, 3):
            sel = [r[1] for r in ok if r[0][col] == v]
            cells.append((lv[v], sum(sel) / len(sel)))
        spread = max(c[1] for c in cells) - min(c[1] for c in cells)
        best_lv = min(cells, key=lambda c: c[1])[0]
        bar = "  ".join(f"{n}={m:>8,.0f}" for n, m in cells)
        print(f"  {name:11} {bar}   极差 {spread:>8,.0f}   最佳 {best_lv}")
    print("\n极差越大 = 该因子影响越大 -> 直接回答 REPORT §4.3「哪些超参真正起作用」")
