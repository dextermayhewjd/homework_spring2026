#!/usr/bin/env bash
# hw2 实验批量运行脚本
#
#   ./run_experiments.sh exp1          # 实验 1：CartPole ×8      （阶段 1 之后）
#   ./run_experiments.sh exp2          # 实验 2：HalfCheetah ×5   （阶段 2 之后）
#   ./run_experiments.sh exp3          # 实验 3：LunarLander ×5   （阶段 3 之后）
#   ./run_experiments.sh exp4          # 实验 4：InvertedPendulum 默认基线
#   ./run_experiments.sh exp4-l9       # 实验 4：L9 正交表调参（9 组）
#
#   PARALLEL=0 ./run_experiments.sh exp1     # 改成串行（默认并行）
#   WANDB_MODE=disabled ./run_experiments.sh exp1   # 不传 wandb
#
# W&B 中固定使用 cs285_hw2 project，并按 exp1/exp2/exp3/exp4 分 group。
# 每次启动的 stdout 保存在 exp_stdout/<suite>/<batch_id>/，同名重跑不会覆盖。
#
# 注意：
#   --no_gpu 是必须的。实测 GPU 比 CPU 慢 1.7 倍 —— 网络只有 4610 个参数，
#   瓶颈在 env.step() 和逐步 get_action 的 host→device 拷贝，不在矩阵乘法。
#   不要用 WANDB_MODE=offline：setup_wandb 里 dir=tempfile.mkdtemp()，
#   离线数据会落到 /tmp 里丢掉。要么 online，要么 disabled。

cd "$(dirname "$0")"

export OMP_NUM_THREADS=1          # 并行时防止各进程的 torch BLAS 互相抢核
PARALLEL=${PARALLEL:-1}

SUITE="${1:-}"
case "$SUITE" in
  exp1) HW2_WANDB_GROUP="exp1_cartpole" ;;
  exp2) HW2_WANDB_GROUP="exp2_halfcheetah" ;;
  exp3) HW2_WANDB_GROUP="exp3_lunarlander_gae" ;;
  exp4) HW2_WANDB_GROUP="exp4_pendulum_tuning" ;;
  exp4-l9) HW2_WANDB_GROUP="exp4_pendulum_L9" ;;
  *)
    sed -n '2,20p' "$0"
    exit 1
    ;;
esac

# PID 避免两个批次在同一秒启动时共用目录。显式传入时则便于外部调度器指定批次名。
HW2_BATCH_ID=${HW2_BATCH_ID:-"$(date +%Y%m%d_%H%M%S)_$$"}
export HW2_WANDB_GROUP HW2_BATCH_ID

OUT="exp_stdout/$SUITE/$HW2_BATCH_ID"
mkdir -p "$OUT"
echo "W&B group: $HW2_WANDB_GROUP"
echo "本批日志: $OUT/"

pids=()
go() {
    local name=""
    for ((i=1; i<=$#; i++)); do
        [ "${!i}" = "--exp_name" ] && { j=$((i+1)); name="${!j}"; }
    done
    echo "▶ $name"
    if [ "$PARALLEL" = 1 ]; then
        uv run src/scripts/run.py --no_gpu "$@" > "$OUT/$name.log" 2>&1 &
        pids+=($!)
    else
        uv run src/scripts/run.py --no_gpu "$@" 2>&1 | tee "$OUT/$name.log"
    fi
}

CP="--env_name CartPole-v0 -n 100"
HC="--env_name HalfCheetah-v4 -n 100 -b 5000 -eb 3000 -rtg --discount 0.95 -lr 0.01"
LL="--env_name LunarLander-v2 --ep_len 1000 --discount 0.99 -n 200 -b 2000 -eb 2000 -l 3 -s 128 -lr 0.001 --use_reward_to_go --use_baseline"

case "$SUITE" in
  exp1)   # 阶段 1 —— 验收：大小 batch 各自的最佳都要收敛到 200
    go $CP -b 1000                --exp_name cartpole
    go $CP -b 1000 -rtg           --exp_name cartpole_rtg
    go $CP -b 1000      -na       --exp_name cartpole_na
    go $CP -b 1000 -rtg -na       --exp_name cartpole_rtg_na
    go $CP -b 4000                --exp_name cartpole_lb
    go $CP -b 4000 -rtg           --exp_name cartpole_lb_rtg
    go $CP -b 4000      -na       --exp_name cartpole_lb_na
    go $CP -b 4000 -rtg -na       --exp_name cartpole_lb_rtg_na
    ;;
  exp2)   # 阶段 2 —— 验收：cheetah_baseline 末尾 Eval > 300
    go $HC                                        --exp_name cheetah
    go $HC --use_baseline -blr 0.01  -bgs 5       --exp_name cheetah_baseline
    # 两个单变量消融：每次只降低一个 baseline 超参数。
    go $HC --use_baseline -blr 0.001 -bgs 5       --exp_name cheetah_baseline_blr0.001
    go $HC --use_baseline -blr 0.01  -bgs 1       --exp_name cheetah_baseline_bgs1
    # PDF Optional：同一个 run 中加回 -na，并每 10 轮记录视频（W&B 需 online）。
    go $HC --use_baseline -blr 0.01 -bgs 5 -na --video_log_freq 10 \
                                                 --exp_name cheetah_baseline_na_video
    ;;
  exp3)   # 阶段 3 —— 验收：最好的一组训练中至少一次 Eval > 150
    for lam in 0 0.95 0.98 0.99 1; do
        go $LL --gae_lambda $lam --exp_name "lunar_lander_lambda$lam"
    done
    ;;
  exp4)   # 阶段 4 调参 —— 见 REPORT §4.1
    # 默认基线（§4.4 的图要用，跑完别删）
    go --env_name InvertedPendulum-v4 -n 100 -b 5000 -eb 1000 --exp_name pendulum
    ;;

  exp4-l9)
    # L9(3^4) 正交表：4 个因子 x 3 水平，9 组估主效应（全因子要 81 组）。
    #   A discount   0.95 / 0.99 / 1.0
    #   B 网络 -l/-s  2,32 / 2,64 / 3,128
    #   C batch -b   500 / 1000 / 2000
    #   D lr         5e-3 / 1e-2 / 2e-2
    # 固定：-rtg -na（实验 1 证实几乎白拿）
    #      --use_baseline --gae_lambda 0.99（实验 3 证实；InvertedPendulum 与
    #      LunarLander 同为 horizon=1000。注意 --gae_lambda 不配 --use_baseline
    #      会被【静默忽略】—— GAE 分支在 critic is not None 里面）
    # 每组预算固定 150K 步（-n = 150000/-b），否则「首次到 1000 的步数」不可比。
    IP="--env_name InvertedPendulum-v4 -eb 1000 -rtg -na --use_baseline --gae_lambda 0.99"
    #        discount  -l  -s     -b   -n    -lr
    l9_run() { go $IP --discount $1 -l $2 -s $3 -b $4 -n $5 -lr $6 --exp_name "pendulum_L9_$7"; }
    l9_run 0.95 2 32   500 300 5e-3 1
    l9_run 0.95 2 64  1000 150 1e-2 2
    l9_run 0.95 3 128 2000  75 2e-2 3
    l9_run 0.99 2 32  1000 150 2e-2 4
    l9_run 0.99 2 64  2000  75 5e-3 5
    l9_run 0.99 3 128  500 300 1e-2 6
    l9_run 1.0  2 32  2000  75 1e-2 7
    l9_run 1.0  2 64   500 300 2e-2 8
    l9_run 1.0  3 128 1000 150 5e-3 9
    ;;
esac

if [ "$PARALLEL" = 1 ] && [ ${#pids[@]} -gt 0 ]; then
    echo "并行运行 ${#pids[@]} 个 run，stdout 在 $OUT/ ..."
    fail=0
    for p in "${pids[@]}"; do wait "$p" || fail=1; done
    [ $fail -eq 0 ] && echo "✅ 全部完成" || { echo "❌ 有 run 失败，查看 $OUT/*.log"; exit 1; }
fi
