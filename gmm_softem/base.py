import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List
from .utils import stable_softmax, weights_cumsum_select


class BaseMatrixSoftEM(ABC):
    def __init__(self, 
                 means: np.ndarray, 
                 covariances: np.ndarray, 
                 weights: np.ndarray, 
                 max_iter: int = 10,
                 tol: float = 1e-4,
                 eps: float = 1e-8,
                 threshold: float = 1.0,
                 verbose: bool = False) -> None:
        """
        初始化 BaseMatrixSoftEM

        参数：
            means: (K, D)
            covariances: shape depends on subclass
            weights: (K,)
            max_iter: 最大迭代次数
            tol: 收敛容忍度
            eps: 数值稳定性参数
            threshold: 组件的保留的阈值，作为主要计算的成分代替全部组件进而加速性能
            verbose: 是否输出详细信息
        """    
        self.means = means  # (K, D)
        self.covariances = covariances  # shape depends on subclass
        self.weights = weights  # (K,)
        assert max_iter > 0, "max_iter should be > 0"
        self.max_iter = max_iter
        self.tol = tol
        self.eps = eps
        assert 1.0 >= threshold > 0.0, "threshold should be <= 1.0 and > 0.0"
        self.truncation_threshold = threshold
        self.verbose = verbose
        self.K = self.means.shape[0]

    def _initialize_theta(self, Y: np.ndarray, method: str = 'auto') -> np.ndarray:
        """
            初始化 theta 参数

        参数：
            Y: 输入数据
            method: 初始化方法 ('auto', 'ones', 'var', 'mad')

        返回：
            初始化的 theta 值
        """    
        if method == 'auto':
            return np.mean((Y[:, None, :] - self.means[None, :, :]) ** 2, axis=1)
        elif method == 'ones':
            return np.ones_like(Y)
        elif method == 'var':
            return np.repeat(np.var(Y, axis=0, keepdims=True), Y.shape[0], axis=0)
        elif method == 'mad':
            return np.tile(np.median(np.abs(Y - np.median(Y, axis=0)), axis=0), (Y.shape[0], 1))
        else:
            raise ValueError(f"Unknown initialization method: {method}")

    def _get_truncated_indices(self, threshold: float = 1.0) -> List[int]:
        """
            self.threshold 决定是否进行组分的截断，保留threshold的比例参与计算
            通常95%的部分能否将K=50 降低到 K=5。
        返回：
            indices 组件对应的原始的索引序列
        """
        threshold = threshold or self.truncation_threshold
        if threshold < 1.0:
            return weights_cumsum_select(self.weights, threshold)
        else:
            return np.arange(len(self.weights)).tolist()    # 不截断

    def _maybe_truncate(self,):
        if self.truncation_threshold < 1.0:
            self._truncate_components(threshold=self.truncation_threshold)

    def _em_theta_step(self,
                       Y: np.ndarray,
                       verbose: bool = False
                       ) -> Tuple[np.ndarray, np.ndarray]:
        """
            em迭代 theta 的在线推理器

        参数：
            Y: 输入数据
            gamma_returned: 是否输出每个样本最终的gamma权重,(N,K)——weights，可用于95%的筛选

        返回：
            初始化的 theta 值
        """
        assert Y.ndim == 2, "Input Y must be a 2D array in em_theta_step"
        self._maybe_truncate()
        N, D= Y.shape

        theta = self._initialize_theta(Y)   # (N, D)
        x_expectation = np.empty_like(Y)  # (N, D)
        gamma = np.empty((N, self.K)) / self.K     # (N, K)

        converged = np.zeros(N, dtype=bool)  # 每个样本是否收敛
        converge_steps = np.full(N, -1)  # 收敛迭代次数记录

        for i in range(self.max_iter):
            # 仅对未收敛样本进行计算
            active_idx = np.where(~converged)[0]
            if len(active_idx) == 0:
                if verbose:
                    print(f"All samples converged at total iteration {i}")
                break

            Y_active = Y[active_idx]
            theta_active = theta[active_idx]

            try:
                # 子类封装矩阵计算
                # Warning: 此处迭代的sigma_post可能出现负数，事实上应该是正数，有些不合理。theta同理也不应该是负数。
                mu_post, sigma_post_diag, covariances = self._compute_cov_terms_batch(theta_active, Y_active)
                # 子类封装 logpdf + log(weights) 的计算
                logpdf = self._compute_weighted_logpdf(Y_active, mu_post, covariances)

                gamma_active = stable_softmax(logpdf, axis=1, keepdims=True)  # shape: (N, K)
                # 计算 E[x] = sum_k γ_nk * μ_k
                x_expectation[active_idx] = np.einsum('nk,nkd->nd', gamma_active, mu_post)  # shape: (N, D)
                # 更新 theta
                expectation = (Y_active[:, None, :] - mu_post)**2 + sigma_post_diag
                theta_new = np.einsum('nk,nkd->nd', gamma_active, expectation)

                # 局部收敛性判断
                delta = np.linalg.norm(theta_new - theta_active, axis=1)
                newly_converged = delta < self.tol

                # 更新收敛状态
                converged[active_idx[newly_converged]] = True
                converge_steps[active_idx[newly_converged]] = i

                # 更新 theta，仅对未收敛样本
                theta[active_idx[~newly_converged]] = theta_new[~newly_converged]

                if verbose and i % 5 == 0:
                    # print(delta)
                    print(f"[SoftEM Step {i}] Δθ avg = {delta.mean():.6f}, {np.sum(converged)} / {N} samples converged")

            except Exception as e:
                print(f"[EM Step Error] Iteration {i} failed: {e}")
                break

        return theta, x_expectation


    @abstractmethod
    def _truncate_components(self, threshold: float) -> None:
        ...


    @abstractmethod
    def _compute_cov_terms_batch(self, theta_iter: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ...

    @abstractmethod
    def _compute_weighted_logpdf(self, Y: np.ndarray, mu_post: np.ndarray, covariances: np.ndarray) -> np.ndarray:
        ...


