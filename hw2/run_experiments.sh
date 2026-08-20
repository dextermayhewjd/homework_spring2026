#!/usr/bin/env bash
# hw2 实验批量运行脚本
#
#   ./run_experiments.sh exp1          # 实验 1：CartPole ×8      （阶段 1 之后）
#   ./run_experiments.sh exp2          # 实验 2：HalfCheetah ×3   （阶段 2 之后）
#   ./run_experiments.sh exp3          # 实验 3：LunarLander ×5   （阶段 3 之后）
#   ./run_experiments.sh exp4          # 实验 4：InvertedPendulum 默认基线
#
#   PARALLEL=0 ./run_experiments.sh exp1     # 改成串行（默认并行）
#   WANDB_MODE=disabled ./run_experiments.sh exp1   # 不传 wandb
#
# 注意：
#   --no_gpu 是必须的。实测 GPU 比 CPU 慢 1.7 倍 —— 网络只有 4610 个参数，
#   瓶颈在 env.step() 和逐步 get_action 的 host→device 拷贝，不在矩阵乘法。
#   不要用 WANDB_MODE=offline：setup_wandb 里 dir=tempfile.mkdtemp()，
#   离线数据会落到 /tmp 里丢掉。要么 online，要么 disabled。

cd "$(dirname "$0")"

export OMP_NUM_THREADS=1          # 并行时防止各进程的 torch BLAS 互相抢核
PARALLEL=${PARALLEL:-1}
OUT=exp_stdout; mkdir -p "$OUT"

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

case "${1:-}" in
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
    go $HC --use_baseline -blr 0.001 -bgs 5       --exp_name cheetah_baseline_blr0.001   # §2.3 的对比实验：降 -blr
    ;;
  exp3)   # 阶段 3 —— 验收：最好的一组训练中至少一次 Eval > 150
    for lam in 0 0.95 0.98 0.99 1; do
        go $LL --gae_lambda $lam --exp_name "lunar_lander_lambda$lam"
    done
    ;;
  exp4)   # 阶段 4 —— 默认基线（§4.4 的图要用，跑完别删）；调参的 run 自己另外加
    go --env_name InvertedPendulum-v4 -n 100 -b 5000 -eb 1000 --exp_name pendulum
    ;;
  *)
    sed -n '2,20p' "$0"; exit 1
    ;;
esac

if [ "$PARALLEL" = 1 ] && [ ${#pids[@]} -gt 0 ]; then
    echo "并行运行 ${#pids[@]} 个 run，stdout 在 $OUT/ ..."
    fail=0
    for p in "${pids[@]}"; do wait "$p" || fail=1; done
    [ $fail -eq 0 ] && echo "✅ 全部完成" || { echo "❌ 有 run 失败，查看 $OUT/*.log"; exit 1; }
fi
