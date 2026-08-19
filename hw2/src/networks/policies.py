"""策略网络。

动作空间有两种,`self.discrete` 在 __init__ 时由 run.py 传入决定,之后不变:

    离散 (CartPole: Discrete(2))         ac_dim = 动作的「个数」
        obs ──▶ logits_net (MLP) ──▶ logits ──▶ Categorical(logits=…)
                                                     └ 内部自动 softmax → probs

    连续 (HalfCheetah: Box(6,))          ac_dim = 动作向量的「维度」
        obs ──▶ mean_net (MLP) ──▶ mean ─────┐
                                              ├──▶ Normal(loc=…, scale=…)
                logstd ──── exp ──▶ std ─────┘
                  ↑ 独立的 nn.Parameter,不吃 obs,与状态无关

关键点
------
1. forward 返回的是「分布对象」,不是 logits、不是动作。
   因为下游有两个需求,一个分布对象同时满足:
       get_action 需要  dist.sample()       抽动作(必须抽样,不能 argmax,否则没有探索且梯度有偏)
       update     需要  dist.log_prob(a)    算 log π(a|s),可求导
   分布对象把「离散 / 连续」的分支收敛在 forward 一处;否则 get_action 和 update 都得各写一遍 if。

2. logstd 存的是 log(标准差),不是标准差。用之前必须 exp:
       - 标准差必须为正,而网络参数是任意实数,存 log 再取指数天然保证正数
       - 忘了 exp 的话,初始 scale = 0.0,Normal 直接报错

3. logstd 不属于 mean_net,所以 mean_net.parameters() 拿不到它。
   __init__ 里用 itertools.chain 把两处参数拼起来交给优化器。
   漏掉的话 logstd 永远不更新,探索噪声固定在 1.0 —— 不报错,但训练上不去。

4. exp 和 softmax 不是并列关系:
       exp     发生在「你的代码里」,构造分布之前   logstd → std
       softmax 发生在「分布对象内部」,你不用写     logits → probs

5. Normal.log_prob(a) 返回逐维度的结果 (batch, ac_dim),
   Categorical.log_prob(a) 返回 (batch,)。
   连续情况在 update 里要对最后一维求和,否则 HalfCheetah 会静默训练失败。

为什么这里要显式构造分布,而平时写网络模块不用
--------------------------------------------
SwiGLU / Attention 这类模块是 tensor -> tensor,网络内部的计算,不涉及概率语义。
LLM 的输出层也不显式构造分布:训练用 F.cross_entropy(分布藏在损失函数里),
推理用 softmax + multinomial(手写采样)。两条路径各干各的,不共享对象。

RL 唯一不同的地方:策略梯度要对「分布」求期望的梯度

    grad J(theta) = E_{a ~ pi_theta} [ grad log pi_theta(a|s) * A(s,a) ]

所以 pi_theta(.|s) 必须是代码里真实存在的对象,而且同一个对象要用两次:
    采样时 dist.sample()  ->  (存下动作)  ->  训练时对「同一个动作」dist.log_prob(a)

分界线不在「LLM vs RL」,而在「是否对一个分布求期望的梯度」。
用 PPO/RLHF 训 LLM 时,LLM 也会变成策略,那时同样要显式算 logprobs。
"""

import itertools
from torch import nn
from torch.nn import functional as F
import torch.distributions as D
from torch import optim

import numpy as np
import torch
from torch import distributions

from infrastructure import pytorch_util as ptu


class MLPPolicy(nn.Module):
    """Base MLP policy, which can take an observation and output a distribution over actions.

    This class should implement the `forward` and `get_action` methods. The `update` method should be written in the
    subclasses, since the policy update rule differs for different algorithms.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        if discrete:
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            parameters = self.logits_net.parameters()
        else:
            self.mean_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
            )
            parameters = itertools.chain([self.logstd], self.mean_net.parameters())

        self.optimizer = optim.Adam(
            parameters,
            learning_rate,
        )

        self.discrete = discrete

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Takes a single observation (as a numpy array) and returns a single action (as a numpy array)."""
        # TODO: implement get_action
        obs = ptu.from_numpy(obs)          # np.ndarray -> torch tensor (放到 ptu.device 上)
        action_distribution = self(obs)    # 调 forward,拿到动作分布
        action = action_distribution.sample()   # 从分布里抽一个动作(不是 argmax!)

        return ptu.to_numpy(action)        # torch tensor -> np.ndarray

    def forward(self, obs: torch.FloatTensor):
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        if self.discrete:
            # TODO: define the forward pass for a policy with a discrete action space.
            # 网络输出当作各动作的 logits,Categorical 内部会自己做 softmax
            logits = self.logits_net(obs)
            return distributions.Categorical(logits=logits)
        else:
            # TODO: define the forward pass for a policy with a continuous action space.
            # 网络输出当作高斯均值;self.logstd 存的是 log(std),要 exp 回标准差
            mean = self.mean_net(obs)
            std = torch.exp(self.logstd)
            return distributions.Normal(loc=mean, scale=std)

    def update(self, obs: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """
        Performs one iteration of gradient descent on the provided batch of data. You don't need to implement this
        method in the base class, but you do need to implement it in the subclass.
        """
        raise NotImplementedError


class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the policy gradient algorithm."""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        """Implements the policy gradient actor update."""
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        # TODO: compute the policy gradient actor loss
        loss = None

        # TODO: perform an optimizer step
        pass

        return {
            "Actor Loss": loss.item(),
        }
