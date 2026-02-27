# dp.py
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import List, Dict, Tuple
import math, random
import numpy as np


# 配置类，包含所有参数
@dataclass
class Config:
    seed: int = 42
    product_names: List[str] = None
    n_friends: int = 20

    horizon_days: int = 7
    shelf_cap: int = 500
    shelf_replenish_per_day: int = 80
    shelf_init: int = 200
    # DP 估计 z 分布用的采样量（越大越稳，越慢）
    z_samples: int = 200_000

    # 评估 episodes
    eval_episodes: int = 20_000


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def lognormal_mu_from_mean(mean: float, sigma: float) -> float:
    return math.log(mean) - 0.5 * sigma * sigma


# 从data.txt读取所有价格，返回float列表
def load_price_data(filepath: str = "data.txt") -> list:
    prices = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                for val in line.split(','):
                    val = val.strip()
                    if val:
                        prices.append(float(val))
    return prices

# 从价格池中采样，返回shape=(num, )的np.ndarray，并加[-50,50]均匀噪声
def sample_prices_with_noise(price_pool, num, rng):
    base = rng.choice(price_pool, size=num, replace=True)
    noise = rng.uniform(-50, 50, size=num)
    return base + noise

#z分布采样：z = max_k(best_sell_k - self_k)，其中 self_k 和 best_sell_k 都是 LogNormal 分布的随机变量
def sample_z_distribution(cfg: Config) -> np.ndarray:
    """
    采样 z = max_k(best_sell_k - self_k) 的经验分布（i.i.d.）
    - self_k ~ LogNormal
    - friends prices ~ LogNormal，best_sell_k = max over agents
    """
    rng = np.random.default_rng(cfg.seed)
    n = len(cfg.product_names)
    agents = 1 + cfg.n_friends
    price_pool = load_price_data()

    # 采样z_samples组，每组(agents, n)个价格
    z_list = []
    for _ in range(cfg.z_samples):
        # 每个agent、每个产品采样一个价格
        prices = sample_prices_with_noise(price_pool, agents * n, rng).reshape((agents, n))
        self_p = prices[0, :]
        best_p = prices.max(axis=0)
        spread = best_p - self_p
        z_val = spread.max()
        z_list.append(z_val)
    return np.array(z_list, dtype=np.float64)


# DP求解：通过 backward induction 计算 V[t,s] 和 theta[t,S_after]
def compute_dp_threshold_policy(cfg: Config, z_samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回：
    - V: shape (H+1, cap+1), V[t,s] = 从第t天开始、当天开始时 shelf=s 的最优期望收益
    - theta: shape (H, cap+1), theta[t,S] = 第t天补货后可用容量为S时的阈值（z >= theta => 满仓做）
      注意：theta 索引用的是 “补货后容量 S”，不是补货前。
    """
    H = cfg.horizon_days
    cap = cfg.shelf_cap
    repl = cfg.shelf_replenish_per_day
    V = np.zeros((H + 1, cap + 1), dtype=np.float64)
    theta = np.zeros((H, cap + 1), dtype=np.float64)

    for t in range(H - 1, -1, -1):
        # 下期价值函数
        V_next = V[t + 1]

        for s in range(cap + 1):
            S_after = min(cap, s + repl)

            # 1) 不操作：V_next[S_after]
            # 2) 梭哈：z*S_after + V_next[0]
            # 贝尔曼最优方程
            base_keep = V_next[S_after]
            base_full_const = V_next[0]

            if S_after == 0:
                V[t, s] = base_keep
                theta[t, S_after] = float("inf")
                continue

            delta = base_keep - base_full_const
            # 阈值：z >= delta / S_after
            th = delta / float(S_after)
            theta[t, S_after] = th

            # 期望：base_keep + E[max(z*S_after - delta, 0)]
            # 用采样近似
            gain = np.maximum(z_samples * float(S_after) - delta, 0.0).mean()
            V[t, s] = base_keep + gain

    return V, theta


# 模拟
def simulate_episode_profit(cfg: Config, theta: np.ndarray, rng: np.random.Generator) -> Tuple[float, float, float]:
    """
    在同一条价格路径上，跑：
    - baseline: 每天补货后满仓做当日最大spread
    - dp_policy: 用阈值决定当日是否满仓
    返回： (profit_dp, profit_baseline, dp_minus_baseline)
    """
    n = len(cfg.product_names)
    agents = 1 + cfg.n_friends
    price_pool = load_price_data()

    shelf_dp = int(cfg.shelf_init)
    shelf_base = int(cfg.shelf_init)

    profit_dp = 0.0
    profit_base = 0.0

    for t in range(cfg.horizon_days):
        shelf_dp = min(cfg.shelf_cap, shelf_dp + cfg.shelf_replenish_per_day)
        shelf_base = min(cfg.shelf_cap, shelf_base + cfg.shelf_replenish_per_day)
        prices = sample_prices_with_noise(price_pool, agents * n, rng).reshape((agents, n))
        self_p = prices[0, :]
        best_p = prices.max(axis=0)
        spread = best_p - self_p
        z = float(spread.max())

        # baseline
        if shelf_base > 0:
            profit_base += float(shelf_base) * z
            shelf_base = 0

        # dp policy
        S_after = shelf_dp
        if S_after > 0:
            th = float(theta[t, S_after])
            if z >= th:
                profit_dp += float(S_after) * z
                shelf_dp = 0

    return profit_dp, profit_base, (profit_dp - profit_base)

#评估
def eval_policy(cfg: Config, theta: np.ndarray) -> Dict[str, float]:
    rng = np.random.default_rng(cfg.seed + 12345)

    dp_total = 0.0
    base_total = 0.0
    diff_total = 0.0
    diffs = []

    for _ in range(cfg.eval_episodes):
        pdp, pbase, diff = simulate_episode_profit(cfg, theta, rng)
        dp_total += pdp
        base_total += pbase
        diff_total += diff
        diffs.append(diff)

    diffs = np.array(diffs, dtype=np.float64)

    return {
        "episodes": float(cfg.eval_episodes),
        "dp_avg_profit": dp_total / cfg.eval_episodes,
        "baseline_avg_profit": base_total / cfg.eval_episodes,
        "dp_minus_baseline_avg": diff_total / cfg.eval_episodes,
        "dp_minus_baseline_p50": float(np.quantile(diffs, 0.50)),
        "dp_minus_baseline_p10": float(np.quantile(diffs, 0.10)),
        "dp_minus_baseline_p90": float(np.quantile(diffs, 0.90)),
    }


def print_threshold_table(cfg: Config, theta: np.ndarray, shelves: List[int] = None):
    if shelves is None:
        shelves = [80, 160, 240, 320, 400, 480, cfg.shelf_cap]

    for t in range(cfg.horizon_days):
        parts = []
        for S in shelves:
            S2 = min(cfg.shelf_cap, S)
            parts.append(f"S={S2:3d}: {theta[t, S2]:8.2f}")
        print(f"Day {t+1:3d}: " + " | ".join(parts))


if __name__ == "__main__":
    cfg = Config(
        seed=42,
        product_names=[f"prod_{i}" for i in range(12)],
        n_friends=20,
        horizon_days=7,  
        shelf_cap=960,
        shelf_replenish_per_day=320,
        shelf_init=320,
        z_samples=200_000,     
        eval_episodes=20_000,  
    )

    set_seed(cfg.seed)

    print("z分布采样中...")
    z = sample_z_distribution(cfg)
    print("\nDP求解中...")
    V, theta = compute_dp_threshold_policy(cfg, z)
    print_threshold_table(cfg, theta)
    print("\n评估中...")
    res = eval_policy(cfg, theta)
    print("\n" + "=" * 80)
    print(f"Episodes                 : {int(res['episodes'])}")
    print(f"DP avg profit            : {res['dp_avg_profit']:,.2f}")
    print(f"Baseline avg profit      : {res['baseline_avg_profit']:,.2f}")
    print(f"DP - Baseline (avg)      : {res['dp_minus_baseline_avg']:,.2f}")
    print(f"DP利润增长率（avg）   : {res['dp_minus_baseline_avg'] / abs(res['baseline_avg_profit']):.2%}")
