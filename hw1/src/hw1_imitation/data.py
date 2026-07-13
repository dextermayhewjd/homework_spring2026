"""Dataset utilities for Push-T."""

from __future__ import annotations

import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

PUSHT_URL = "https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip"
ZARR_RELATIVE_PATH = Path("pusht") / "pusht_cchi_v7_replay.zarr"


@dataclass(frozen=True)
class Normalizer:
    """Feature-wise normalizer for states and actions.

    做的事:对 state 和 action 做标准化(z-score 归一化),公式就是
    (x - 均值) / 标准差,处理后每个特征的分布近似「均值≈0、标准差≈1」。
    为什么:神经网络对输入的数值尺度敏感,若一个特征范围 0~1、另一个 0~10000,
    训练会不稳定;归一化让各特征尺度一致,训练更快更稳。
    "Feature-wise"(逐特征):统计时用 axis=0,即按列/按维度分别算 mean 和 std,
    每个特征用自己的 μ、σ 归一化(D 维特征 → D 个 mean、D 个 std)。
    """

    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray

    @staticmethod
    def _safe_std(std: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        # 数值安全兜底:给标准差设一个下限 eps(1e-6)。
        # np.maximum(std, eps) = 逐元素取 std 与 eps 的较大者。
        # 目的是防止除以 0:若某特征在数据里是常数,则它的 std=0,
        # 归一化 (x - mean) / 0 会得到 NaN/inf 让训练崩掉。
        # 于是把过小(接近 0)的 std 替换成 1e-6。eps 在数值计算里就是
        # 「防止除零/取对数出错的极小量」。
        return np.maximum(std, eps)

    @classmethod
    def from_data(cls, states: np.ndarray, actions: np.ndarray) -> "Normalizer":
        # 从一批数据里统计出每个特征的 mean 和 std 存起来备用。
        # ndarray 自带 .mean()/.std() 方法;axis=0 表示沿样本方向统计,
        # 得到「每个特征各自的」统计量(长度 = 特征维度 D)。
        # 注:numpy 的 .std() 默认 ddof=0(总体标准差,除以 N);归一化场景无所谓。
        state_mean = states.mean(axis=0)
        state_std = cls._safe_std(states.std(axis=0))
        action_mean = actions.mean(axis=0)
        action_std = cls._safe_std(actions.std(axis=0))
        return cls(state_mean, state_std, action_mean, action_std)

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        # 核心归一化公式:(x - μ) / σ
        return (state - self.state_mean) / self.state_std

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        return (action - self.action_mean) / self.action_std

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        # 反归一化(还原):x * σ + μ,是 normalize 的逆运算。
        # 网络输出的是「归一化空间」的动作,真正喂给环境执行前要还原回原始尺度。
        return action * self.action_std + self.action_mean


def download_pusht(dataset_dir: Path) -> Path:
    """Download and extract the Push-T dataset if needed.

    Returns the path to the extracted Zarr dataset.

    做的事:确保 Push-T 数据集在本地就绪,并返回解压后的 Zarr 数据路径。
    整个函数是「幂等 + 带缓存」的:重复调用不会重复下载/解压,只做缺失的步骤,
    所以可以放心地每次训练前都调一遍。
    """

    # exist_ok=True:目录已存在也不报错;parents=True:父目录不存在则一并创建。
    dataset_dir.mkdir(parents=True, exist_ok=True)
    zarr_path = dataset_dir / ZARR_RELATIVE_PATH
    # 缓存判断①:如果最终的 Zarr 数据已经解压好了,直接返回,啥都不用做。
    if zarr_path.exists():
        return zarr_path

    zip_path = dataset_dir / "pusht.zip"
    # 缓存判断②:zip 还没下过才去网上下载(避免重复下载几百 MB 的大文件)。
    # urlretrieve(url, 目标路径):把远端文件保存到本地。
    if not zip_path.exists():
        urllib.request.urlretrieve(PUSHT_URL, zip_path)

    # 解压 zip 到 dataset_dir;with 语句保证用完自动关闭文件句柄。
    # 解压后就会在 dataset_dir 下得到上面的 zarr_path。
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dataset_dir)

    return zarr_path


def load_pusht_zarr(zarr_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从 Zarr 文件里把 Push-T 数据一次性读进内存。

    返回三个数组:
      - states:所有时间步的状态,形状 (T, state_dim)
      - actions:所有时间步的动作,形状 (T, action_dim)
      - episode_ends:每条轨迹(episode)的「结束下标」,见 build_valid_indices。
    这里 T 是把所有 episode「首尾拼接」后的总时间步数(它们共享一条大数组)。
    """
    # zarr 是一种适合大数组的分块存储格式;mode="r" 只读打开。
    root = zarr.open(zarr_path, mode="r")
    # [:] 表示把整个数组从磁盘读进内存;np.asarray + dtype 统一成训练要用的类型。
    # 状态/动作用 float32(省显存、够精度),下标用 int64(整数索引)。
    states = np.asarray(root["data"]["state"][:], dtype=np.float32)
    actions = np.asarray(root["data"]["action"][:], dtype=np.float32)
    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    return states, actions, episode_ends


def build_valid_indices(episode_ends: np.ndarray, chunk_size: int) -> np.ndarray:
    """找出所有「合法的滑动窗口起点 t」,即从 t 取 chunk_size 步动作时
    不会跨越 episode 边界的那些 t。

    背景:所有 episode 被拼成一条大数组,episode_ends 记录每条的结束下标(不含)。
    例如 episode_ends = [50, 120, 200] 表示三条 episode 占用的下标区间为:
      [0, 50)、[50, 120)、[120, 200)。
    一个训练样本会取 actions[t : t + chunk_size],所以要求 t + chunk_size <= end,
    否则这段动作就「串」到下一条
    episode 里去了,是错误的监督信号。

    返回值是什么(重要):返回的是一串「合法起点 t」的一维整数数组,例如
    [0,1,2, 5,6, ...],而**不是**把窗口里的数据切出来存下来。真正按起点去切
    (states[t], actions[t : t+chunk_size]) 的动作被推迟到 PushtChunkDataset.__getitem__,
    用时才现切。好处:① 省内存(相邻窗口大量重叠,只存一个 int 起点几乎不占空间);
    ② 契合 PyTorch Dataset 的惰性取数(__len__ = len(indices),__getitem__ 按需现算)。
    """
    # 每条 episode 的起点 = 上一条的终点;第一条从 0 开始。
    # episode_ends[:-1] 去掉最后一个,再在前面拼个 0 → 得到与 ends 一一对应的 starts。
    #
    # np.concatenate 用法:参数要传「一个序列(元组/列表)」,里面装各段一维数组,
    # 它会把这些段首尾拼接成一条数组。注意这里是双层括号 (( [0], episode_ends[:-1] )):
    # 外层是 concatenate 的参数元组,内层 [0] 是第一段。
    # 形状推导(设共 K 条 episode,episode_ends 形状 (K,)):
    #   [0]                -> 长度 1
    #   episode_ends[:-1]  -> 形状 (K-1,)
    #   拼接后 starts       -> 形状 (K,),即与 episode_ends 同长,才能后面 zip 到一起。
    starts = np.concatenate(([0], episode_ends[:-1]))
    indices: list[int] = []
    # 遍历每条 episode 的 [start, end) 区间;strict=True 要求两者长度一致(更安全)。
    for start, end in zip(starts, episode_ends, strict=True):
        # last_start:在本 episode 内能放下一整段 chunk 的「最后一个」合法起点。
        # 因为需要 t + chunk_size <= end,所以 t 最大只能到 end - chunk_size。
        last_start = end - chunk_size
        # 若连一段都放不下(episode 太短),跳过这条 episode。
        if last_start < start:
            continue
        # 合法起点是 start, start+1, ..., last_start(闭区间),故 range 到 last_start+1。
        # range 的步长默认是 1,所以生成的是连续整数 start, start+1, ..., last_start,
        # 正好对应滑动窗口「一格一格往右挪」的每个起点。
        # 用 extend(不是 append):extend 会把 range 里的元素「逐个」拆开追加,得到
        # [0,1,2,...];若用 append(range(...)) 则会把整个 range 当作「一个」元素塞进去。
        indices.extend(range(start, last_start + 1))
    return np.asarray(indices, dtype=np.int64)


class PushtChunkDataset(Dataset):
    """Dataset of (state, action_chunk) pairs using a sliding window.

    这是整条数据管线的「收尾」:把前面读到的 states/actions,配合 build_valid_indices
    算出的合法起点,组装成一个个 (状态, 未来 chunk_size 步动作) 训练样本。
    继承 torch.utils.data.Dataset,只需实现两个「协议方法」,就能交给 DataLoader
    自动做打乱、分批(batch)、多进程加载:
      - __len__     -> 样本总数
      - __getitem__ -> 按下标取第 idx 个样本
    设计要点:__init__ 只存「起点索引」(不预切数据),真正的切片延后到 __getitem__
    现切 —— 省内存 + 惰性取数,和 build_valid_indices 的说明一脉相承。
    """

    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        episode_ends: np.ndarray,
        chunk_size: int,
        normalizer: Normalizer | None = None,
    ) -> None:
        # 直接持有整块 states/actions 的引用(不复制),省内存;chunk_size 决定动作窗口长度。
        self.states = states
        self.actions = actions
        self.chunk_size = chunk_size
        # normalizer 可选:传入则在取样本时对 state/action 做归一化;None 则原样返回。
        self.normalizer = normalizer
        # 预先算好所有「合法滑动窗口起点」(一维整数数组),__getitem__ 用它把 idx 映射到 t。
        self.indices = build_valid_indices(episode_ends, chunk_size)

    def __len__(self) -> int:
        # 样本总数 = 合法起点的个数(DataLoader 靠它知道一个 epoch 要遍历多少个样本)。
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # idx 是 0..len-1 的「第几个样本」;先经 indices 映射成大数组里的真实起点 t。
        # int(...) 把 numpy 整数转成 Python int,避免后续切片/索引出现类型上的小意外。
        t = int(self.indices[idx])
        state = self.states[t]                              # 当前时刻的状态,形状 (state_dim,)
        action_chunk = self.actions[t : t + self.chunk_size]  # 未来 chunk_size 步动作,(chunk_size, action_dim)

        # 有 normalizer 就把输入/标签都归一化到统一尺度(见 Normalizer 说明)。
        if self.normalizer is not None:
            state = self.normalizer.normalize_state(state)
            action_chunk = self.normalizer.normalize_action(action_chunk)

        # 转成 torch.Tensor 并统一成 float32(网络要的类型);from_numpy 会共享内存,.float() 保证 dtype。
        return (
            torch.from_numpy(state).float(),
            torch.from_numpy(action_chunk).float(),
        )
