"""Train and evaluate a Push-T imitation policy."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro
import wandb
from torch.utils.data import DataLoader

from hw1_imitation.data import (
    Normalizer,
    PushtChunkDataset,
    download_pusht,
    load_pusht_zarr,
)
from hw1_imitation.model import build_policy, PolicyType
from hw1_imitation.evaluation import Logger, evaluate_policy

LOGDIR_PREFIX = "exp"


@dataclass
class TrainConfig:
    # The path to download the Push-T dataset to.
    data_dir: Path = Path("data")

    # The policy type -- either MSE or flow.
    policy_type: PolicyType = "mse"
    # The number of denoising steps to use for the flow policy (has no effect for the MSE policy).
    flow_num_steps: int = 10
    # The action chunk size.
    chunk_size: int = 8

    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.0
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    # The number of epochs to train for.
    num_epochs: int = 400
    # How often to run evaluation, measured in training steps.
    eval_interval: int = 10_000
    num_video_episodes: int = 5
    video_size: tuple[int, int] = (256, 256)
    # How often to log training metrics, measured in training steps.
    log_interval: int = 100
    # Random seed.
    seed: int = 42
    # WandB project name.
    wandb_project: str = "hw1-imitation"
    # Experiment name suffix for logging and WandB.
    exp_name: str | None = None


def parse_train_config(
    args: list[str] | None = None,
    *,
    defaults: TrainConfig | None = None,
    description: str = "Train a Push-T MLP policy.",
) -> TrainConfig:
    defaults = defaults or TrainConfig()
    return tyro.cli(
        TrainConfig,
        args=args,
        default=defaults,
        description=description,
    )


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def config_to_dict(config: TrainConfig) -> dict[str, Any]:
    data = asdict(config)
    for key, value in data.items():
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def run_training(config: TrainConfig) -> None:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    zarr_path = download_pusht(config.data_dir)
    states, actions, episode_ends = load_pusht_zarr(zarr_path)
    normalizer = Normalizer.from_data(states, actions)

    dataset = PushtChunkDataset(
        states,
        actions,
        episode_ends,
        chunk_size=config.chunk_size,
        normalizer=normalizer,
    )

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )

    model = build_policy(
        config.policy_type,
        state_dim=states.shape[1],
        action_dim=actions.shape[1],
        chunk_size=config.chunk_size,
        hidden_dims=config.hidden_dims,
    ).to(device)

    # torch.compile 编译网络前向以加速训练步(首次前向有一次编译预热,之后更快)。
    # 编译的是 self.net —— 两个策略的 compute_loss / sample_actions 都走它,通用。
    # 做一次探测前向触发编译;若本机编译后端不可用(如缺 CUDA 工具链)则回退未编译,保证能跑。
    net_in_features = model.net[0].in_features
    _compiled_net = torch.compile(model.net)
    try:
        with torch.no_grad():
            _compiled_net(torch.zeros(2, net_in_features, device=device))
        model.net = _compiled_net
        print("torch.compile enabled")
    except Exception as exc:  # noqa: BLE001
        print(f"torch.compile unavailable, using eager mode ({type(exc).__name__})")

    exp_name = f"seed_{config.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if config.exp_name is not None:
        exp_name += f"_{config.exp_name}"
    log_dir = Path(LOGDIR_PREFIX) / exp_name
    wandb.init(
        project=config.wandb_project, config=config_to_dict(config), name=exp_name
    )
    logger = Logger(log_dir)

    # 优化器:Adam(Kingma 2014)根据梯度更新 model 的所有可训练参数(即 self.net 的权重)。
    # model.parameters() 由 nn.Module 自动收集;lr / weight_decay 来自 config。
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    step = 0  # 全局训练步数(跨 epoch 累加),用于控制 log / eval 的节奏
    last_log_time = time.perf_counter()  # 用于估算训练速度(步/秒)
    for epoch in range(config.num_epochs):
        for state, action_chunk in loader:  # DataLoader 每次给一个 batch
            model.train()  # 训练模式(evaluate_policy 会切成 eval,这里切回来)

            # 数据搬到 model 所在设备(CPU/GPU 必须一致)。
            state = state.to(device)
            action_chunk = action_chunk.to(device)

            # —— 核心四步:算 loss → 清梯度 → 反传 → 更新参数 ——
            loss = model.compute_loss(state, action_chunk)  # 你在 model.py 写的
            optimizer.zero_grad()  # 清掉上一步的梯度(PyTorch 默认累加)
            loss.backward()        # 反向传播,算出每个参数的梯度
            optimizer.step()       # 用梯度更新参数(self.net 权重在这里被改)

            # 定期记录训练 loss + 训练速度(logger.log 会同时写 CSV + wandb)。
            if step % config.log_interval == 0:
                now = time.perf_counter()
                # 步/秒:自上次记录以来的平均速度(step 0 还没区间,记 0)。
                sps = 0.0 if step == 0 else config.log_interval / (now - last_log_time)
                last_log_time = now
                logger.log(
                    {"train/loss": loss.item(), "train/steps_per_sec": sps}, step=step
                )
                print(
                    f"epoch {epoch}  step {step}  loss {loss.item():.4f}  ({sps:.0f} steps/s)"
                )

            # 定期在真实 Push-T 环境里评测(跑 rollout、录视频、存 checkpoint)。
            # 跳过 step 0(此时模型还没训,评测又慢),训练结束后再补一次最终评测。
            if step > 0 and step % config.eval_interval == 0:
                evaluate_policy(
                    model,
                    normalizer,
                    device,
                    chunk_size=config.chunk_size,
                    video_size=config.video_size,
                    num_video_episodes=config.num_video_episodes,
                    flow_num_steps=config.flow_num_steps,
                    step=step,
                    logger=logger,
                )

            step += 1

    # 训练结束后再评测一次,确保拿到最终模型的成绩(供打分)。
    evaluate_policy(
        model,
        normalizer,
        device,
        chunk_size=config.chunk_size,
        video_size=config.video_size,
        num_video_episodes=config.num_video_episodes,
        flow_num_steps=config.flow_num_steps,
        step=step,
        logger=logger,
    )

    logger.dump_for_grading()


def main() -> None:
    config = parse_train_config()
    run_training(config)


if __name__ == "__main__":
    main()
