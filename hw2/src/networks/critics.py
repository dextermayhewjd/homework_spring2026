"""Value network（critic / baseline）。

形状后缀约定（详见 SHAPES.md「本仓库采用的约定」）：
    B = batch，拍平后的时间步总数
    O = ob_dim，观测维度

它和 actor 的根本区别：critic 是**普通的监督学习回归器** ——
输入状态、输出一个数、和蒙特卡洛回报比、算 MSE。
所以它的 loss 是**真 loss**（该降，能拿来 debug），
不像 actor 的代理损失数值无意义。见 VARIANCE.md 的四技巧对照表。
"""

import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from infrastructure import pytorch_util as ptu


class ValueCritic(nn.Module):
    """Value network, which takes an observation and outputs a value for that observation."""

    def __init__(
        self,
        ob_dim: int,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        self.network = ptu.build_mlp(
            input_size=ob_dim,
            output_size=1,
            n_layers=n_layers,
            size=layer_size,
        ).to(ptu.device)

        self.optimizer = optim.Adam(
            self.network.parameters(),
            learning_rate,
        )

    def forward(self, obs_BO: torch.Tensor) -> torch.Tensor:
        """(B, O) -> (B,)

        network 的 output_size=1，原始输出是 (B, 1)。那根长度为 1 的轴是
        build_mlp 需要一个 output_size 参数才产生的记账产物，不携带信息 ——
        value function 概念上就是「一个状态一个标量」。在这里 squeeze 掉，
        对外契约统一成 (B,)，两个调用点（update 的 MSE、_estimate_advantage
        减 baseline）都不必再想这根轴。

        不 squeeze 的后果：MSE 会把 (B,1) 和 (B,) 广播成 (B,B)，
        不报错、只有一条 UserWarning，critic 退化成输出常数。见 SHAPES.md。
        """
        values_B1 = self.network(obs_BO)
        return values_B1.squeeze(-1)

    def update(self, obs: np.ndarray, q_values: np.ndarray) -> dict:
        """一步监督回归：让 V_φ(s) 逼近从该状态实际观测到的折扣回报。

        obs      (B, O)  状态
        q_values (B,)    该状态出发实际拿到的折扣回报。**蒙特卡洛估计，无网络参与**,
                         由 _calculate_q_vals 从真实奖励序列算出。
                         注意它的内容取决于 use_reward_to_go：只有 -rtg 那一版才是
                         PDF 式 (13) 中 V^π(s_t) 的合法目标（求和从 t 开始）。
                         不带 -rtg 时目标含了 t 之前的奖励，critic 会安静地学错东西。
        """
        obs_BO = ptu.from_numpy(obs)
        q_values_B = ptu.from_numpy(q_values)   # requires_grad=False，天然是常数标签，无需 detach

        # TODO: compute the loss using the observations and q_values
        values_B = self(obs_BO)
        # values_B 来自 critic 前向、q_values_B 来自 MC 回报 —— 两条独立路径首次相遇，钉住
        assert values_B.shape == q_values_B.shape, (values_B.shape, q_values_B.shape)
        loss = F.mse_loss(values_B, q_values_B)

        # TODO: perform an optimizer step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "Baseline Loss": loss.item(),
        }