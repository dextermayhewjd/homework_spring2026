"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""
    '''
    首先需要考虑这里policy的输入和输出
    输入的size是chunk size 步的action
    那么这里就需要考虑chunk size和action dim
    
    '''
    
    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,# 注意这里的chunk size 是action的chunk size， 连续的t个action
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        # 输出宽度:一次要吐出 chunk_size 步、每步 action_dim 维 → 拉平成一个向量。
        output_dim = chunk_size * action_dim

        # 维度链:把「输入宽 + 各隐藏层宽」串起来,例如 [state_dim, 256, 256, 256]。
        # *hidden_dims 是把元组拆开摊进列表。
        dims = [state_dim, *hidden_dims]

        # 相邻两两配对建 Linear+ReLU:(state_dim→256)(256→256)(256→256)...
        # zip(dims[:-1], dims[1:]) = 每层的 (入宽, 出宽);和 data.py 里的相邻配对同一套路。
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())

        # 输出层:最后一个隐藏层 → output_dim;后面不接 ReLU(动作可正可负)。
        layers.append(nn.Linear(dims[-1], output_dim))

        # 打包成顺序网络,存成属性供前向使用;*layers 把列表拆成多个参数。
        self.net = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        # state 是 2 维 (B, state_dim);action_chunk 是 3 维 (B, chunk_size, action_dim)。
        B = state.shape[0]
        # 前向:得到拉平的预测动作 (B, chunk_size*action_dim)。
        predicted = self.net(state)
        # reshape 回 (B, chunk_size, action_dim),和 action_chunk 形状对齐才能逐元素比。
        # self.chunk_size / self.action_dim 是 BasePolicy.__init__ 存好的。
        predicted = predicted.reshape(B, self.chunk_size, self.action_dim)
        # MSE = 逐元素平方差的平均(就是手算小例子那个),返回一个标量 loss。
        loss = nn.functional.mse_loss(predicted, action_chunk)
        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        
        # MSE 版:一次前向直接出整段动作,num_steps 用不到(那是 flow 的迭代步数)。
        # 输入 state 是 (B, state_dim)(评测时 B=1,已由 unsqueeze(0) 补好 batch 维)。
        B = state.shape[0]
        predicted = self.net(state)                                     # (B, chunk_size*action_dim)
        predicted = predicted.reshape(B, self.chunk_size, self.action_dim)  # (B, chunk_size, action_dim)
        return predicted  # 直接返回动作段(不算 loss);外部已包 torch.no_grad() 


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        # 拉平后的动作维度,后面反复用到。
        self.action_flat_dim = chunk_size * action_dim
        # ★不同点①:网络输入不再只有 state,而是 [观测 o, 带噪动作 a_{t,τ}(拉平), 流时间 τ] 拼在一起,
        #   对应 slide 15 的 v_θ(o_t, a_{t,τ}, τ)。输入宽度 = state_dim + chunk*action_dim + 1(+1 是流时间标量)。
        input_dim = state_dim + self.action_flat_dim + 1
        # 输出是「速度」,和拉平动作同宽(每个动作分量对应一个移动方向)。
        output_dim = self.action_flat_dim

        # MLP 的搭法和 MSEPolicy 完全一样,只是进出口宽度变了。
        dims = [input_dim, *hidden_dims]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def _predict_velocity(
        self, state: torch.Tensor, a_tau: torch.Tensor, tau: torch.Tensor
    ) -> torch.Tensor:
        """速度网络 v_θ(o, a_{t,τ}, τ):给定(观测, 半路动作 a_{t,τ}, 流时间 τ)预测速度。

        ───────────── 维度速查 ─────────────
        输入:
          state = o        (B, state_dim)            观测/状态
          a_tau = a_{t,τ}  (B, chunk, action_dim)    连线上 τ 处的「半路动作段」
          tau   = τ        (B,)                       每个样本一个流时间标量 ∈[0,1]
        输出:
          返回             (B, chunk, action_dim)    速度,和动作段完全同形状
        ──────────────────────────────────
        ★这里用 τ(tau)表示「流时间 ∈ [0,1]」,别和数据里的「轨迹时间步 t」(第几步)混淆——
          这正是 slide 15 特意把流时间改名成 τ 的原因。
        """
        B = state.shape[0]                                   # B = 批大小(从 state 的第 0 维取)
        # a_tau 是三维动作段,先拍平成二维好喂给 Linear(只吃二维 (B, 特征))。
        # action_flat_dim = chunk*action_dim。
        a_flat = a_tau.reshape(B, self.action_flat_dim)      # (B, chunk, action_dim) -> (B, chunk*action_dim)
        # tau 是一维 (B,),升成二维列向量,才能和下面的 state/a_flat 沿特征维拼接。
        tau_col = tau.reshape(B, 1)                          # (B,) -> (B, 1)
        # ★不同点②:三样东西沿特征维(dim=1)拼成网络输入。
        # 宽度 = state_dim + chunk*action_dim + 1。
        #   [ state | a_flat | tau_col ]
        net_in = torch.cat([state, a_flat, tau_col], dim=1)  # -> (B, state_dim + chunk*action_dim + 1)
        # 过 MLP:输入宽 state_dim+chunk*action_dim+1 -> 输出 output_dim = action_flat_dim = chunk*action_dim。
        v_flat = self.net(net_in)                            # -> (B, chunk*action_dim)
        # 把拍平的速度折回动作段形状,和 a_tau / v_target 对齐,才能做逐元素运算。
        return v_flat.reshape(B, self.chunk_size, self.action_dim)  # (B, chunk*action_dim) -> (B, chunk, action_dim)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        # ★不同点③:loss 比的是「速度」,照搬 slide 15 训练框(a_{t,0}=噪声, a_t=真实动作)。
        B = state.shape[0]
        a1 = action_chunk                                    # a_t:数据端(τ=1)= 真实动作 (B, chunk, action_dim)
        a0 = torch.randn_like(a1)                            # a_{t,0}:噪声端(τ=0)~ N(0,I),同形状
        # ← 训练时 τ 从这来:随机抽 τ ~ U(0,1),每个样本各一个(slide 15: sample τ⁽ʲ⁾ ~ p(τ))。
        tau = torch.rand(B, device=a1.device)
        tau_b = tau.reshape(B, 1, 1)                          # 广播到动作形状,用于插值
        a_tau = (1.0 - tau_b) * a0 + tau_b * a1              # a_{t,τ}=τ·a1+(1-τ)·a0:连线上 τ 处的点
        v_target = a1 - a0                                   # 目标速度 = a_t - a_{t,0}(直线对 τ 的导数)
        v_pred = self._predict_velocity(state, a_tau, tau)   # 网络预测速度 v_θ(o, a_{t,τ}, τ)
        # 仍是 MSE(= slide 的 ‖v - (a1-a0)‖²),只不过比的是速度而不是动作本身。
        return nn.functional.mse_loss(v_pred, v_target)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        # ★不同点④:不是一次前向出结果,而是从噪声出发按 Euler 积分迭代 num_steps 次(slide 14 采样框)。
        B = state.shape[0]
        # 从纯噪声起步(对应 τ=0 的 a_{t,0})。
        a = torch.randn(B, self.chunk_size, self.action_dim, device=state.device)
        dt = 1.0 / num_steps                                 # Δτ:把流时间 [0,1] 均分成 num_steps 步
        for i in range(num_steps):
            # ← 采样时 τ 从这来:不是随机,而是按顺序推进 0, Δτ, 2Δτ, ...(slide 14: for t∈{0,Δt,...})。
            # torch.full((形状), 填充值):造一个该形状、每格都填成同一个值的张量。
            # 这里形状 (B,)、值 i*dt,即整批 B 个样本此刻共享同一个流时间 τ=i*dt。
            #   例:B=4, num_steps=5 -> dt=0.2,循环里 tau 依次是
            #     第0步 [0.0,0.0,0.0,0.0]  第1步 [0.2,0.2,0.2,0.2]  第2步 [0.4,...]  ... 第4步 [0.8,...]
            #   行内全相同(batch 共享 τ),行间从 0 往 1 递增(整批同步推进)。
            #   对比训练 compute_loss 用的是 torch.rand(B) —— 每样本一个各不相同的随机 τ。
            tau = torch.full((B,), i * dt, device=state.device)
            v = self._predict_velocity(state, a, tau)          # 预测该往哪挪 v_θ(o, a_{t,τ}, τ)
            a = a + dt * v                                     # a_{τ+Δτ} ← a_τ + v·Δτ(欧拉法前进一步)
        return a                                             # 迭代到 τ=1,a 就是流出来的动作段


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
