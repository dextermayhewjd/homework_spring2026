from typing import Sequence, Callable, Tuple, Optional

import torch
from torch import nn

import numpy as np

from infrastructure import pytorch_util as ptu


class DQNAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        make_critic: Callable[[Tuple[int, ...], int], nn.Module],
        make_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_lr_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        discount: float,
        target_update_period: int,
        use_double_q: bool = False,
        clip_grad_norm: Optional[float] = None,
    ):
        super().__init__()

        self.critic = make_critic(observation_shape, num_actions)
        self.target_critic = make_critic(observation_shape, num_actions)
        self.critic_optimizer = make_optimizer(self.critic.parameters())
        self.lr_scheduler = make_lr_schedule(self.critic_optimizer)

        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.discount = discount
        self.target_update_period = target_update_period
        self.clip_grad_norm = clip_grad_norm
        self.use_double_q = use_double_q

        self.critic_loss = nn.MSELoss()

        self.update_target_critic()

    def get_action(self, observation: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Epsilon-greedy action selection (default epsilon=0 for deterministic/greedy policy).
        """
        observation = ptu.from_numpy(np.asarray(observation))[None]

        # TODO(Section 2.4): get the action from the critic using an epsilon-greedy strategy
        # 采用讲义口径：π(a|s) = 1-ε 若 a = argmax Q，否则 ε/(|A|-1)。
        # 注意 otherwise 那支【排除】贪心动作，所以两支都要先知道贪心动作是谁。
        # critic(observation) → (1, num_actions)，最后一维是动作轴，argmax 消掉它得到 (1,)。
        greedy_action = self.critic(observation).argmax(dim=-1)

        if np.random.random() < epsilon:
            # 在剩下的 |A|-1 个动作里均匀抽。用「贪心下标 + [1, n) 的偏移，再取模」
            # 一次抽中，不需要拒绝采样重试。
            # device= 不能省：greedy_action 在 GPU 上，randint 默认建在 CPU，
            # 两者相加会抛 "Expected all tensors to be on the same device"。
            offset = torch.randint(1, self.num_actions, (1,), device=greedy_action.device)
            action = (greedy_action + offset) % self.num_actions
        else:
            action = greedy_action
        # ENDTODO

        return ptu.to_numpy(action).squeeze(0).item()

    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> dict:
        """Update the DQN critic, and return stats for logging."""
        (batch_size,) = reward.shape

        # Compute target values
        with torch.no_grad():
            # TODO(Section 2.4): compute target values
            # T0: 课件第 3 步  y_j = r_j + γ(1-d_j)·max_{a'} Q_θ̄(s'_j, a'_j)
            # 后缀 B = batch_size，A = 动作轴长度（DQN 下即 num_actions）。

            # 下一状态、所有动作的 Q 值。用 target 网络 Q_θ̄，不是在线网络。
            next_qa_values_BA: torch.Tensor = self.target_critic(next_obs)

            if self.use_double_q:
                # TODO(Section 2.5): implement double-Q target action selection
                next_action_B: torch.Tensor = self.critic(next_obs).argmax(dim=-1)
            else:
                # 选 a'：哪个动作的 Q 最大。argmax 消掉动作轴，BA -> B，值即动作编号。
                next_action_B: torch.Tensor = next_qa_values_BA.argmax(dim=-1)

            # 按列号取值：每行从 A 个 Q 里挑出 next_action_B 指定的那一个。BA + B -> B。
            # gather 要求 index 与 input 同维，故先 unsqueeze(1) 成 (B,1)，取完再 squeeze 回 (B,)。
            next_q_values_B = torch.gather(
                next_qa_values_BA, 1, next_action_B.unsqueeze(1)
            ).squeeze(1)
            assert next_q_values_B.shape == (batch_size,), next_q_values_B.shape

            # done 是 torch.bool，必须先 .float()——`1 - done` 会直接抛 RuntimeError。
            # 终止之后没有未来回报，(1-d) 把 bootstrap 项掐掉，y 退化成纯 reward。
            target_values_B = reward + self.discount * (1 - done.float()) * next_q_values_B
            assert target_values_B.shape == (batch_size,), target_values_B.shape
            # ENDTODO

        # TODO(Section 2.4): train the critic with the target values
        # 在线网络 Q_θ 对【当前】状态前向。和上半段结构相同，只是换了网络和观测。
        qa_values_BA: torch.Tensor = self.critic(obs)
        # 按列号取值，但列号是【实际执行过的】动作（buffer 里存的），不是 argmax。
        # action 从 buffer 出来就是 int64，gather 要的正是 int64，不用转。
        q_values_B = torch.gather(
            qa_values_BA, dim=1, index=action.unsqueeze(1)
        ).squeeze(1)
        assert q_values_B.shape == (batch_size,), q_values_B.shape
        # self.critic_loss 是 __init__ 里建好的 nn.MSELoss()：平方 + 对 B 取平均，
        # 把 (B,) 的逐样本误差收成 0 维标量，backward() 才有唯一的求导目标。
        loss = self.critic_loss(q_values_B, target_values_B)
        # ENDTODO

        self.critic_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            self.critic.parameters(), self.clip_grad_norm or float("inf")
        )
        self.critic_optimizer.step()

        self.lr_scheduler.step()

        return {
            "critic_loss": loss.item(),
            "q_values": q_values_B.mean().item(),
            "target_values": target_values_B.mean().item(),
            "grad_norm": grad_norm.item(),
        }

    def update_target_critic(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def update(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
        step: int,
    ) -> dict:
        """
        Update the DQN agent, including both the critic and target.
        """
        # TODO(Section 2.4): update the critic, and the target if needed
        # 课件第 4 步：用 target 值训练在线网络。
        # 返回的是日志用的标量字典，原样往上传给 run_dqn.py 的 update_info。
        critic_stats = self.update_critic(obs, action, reward, next_obs, done)

        # 课件第 5 步：每 target_update_period 步把 θ 拷给 θ̄。
        # step 由 run_dqn.py 的训练循环给，agent 只读不改。
        if step % self.target_update_period == 0:
            self.update_target_critic()
        # ENDTODO

        return critic_stats
