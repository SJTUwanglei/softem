from .base import BaseMatrixSoftEM
import numpy as np
from .utils import safe_clip
from typing import Tuple


class DiagMatrixSoftEM(BaseMatrixSoftEM):
    def _truncate_components(self, threshold: float = 1.0) -> None:
        indices = self._get_truncated_indices(threshold)
        self.means = self.means[indices]
        self.weights = self.weights[indices]
        self.covariances = self.covariances[indices]
        self.K = len(indices)

    def _compute_cov_terms_batch(self, theta_iter: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算条件均值和协方差（对角结构）

        参数：
            theta_iter: (N, D) 当前样本的条件方差估计
            Y: (N, D) 输入样本

        返回：
            mu_post: (N, K, D) 条件均值
            sigma_post_diag: (N, K, D) 条件协方差（对角）
            covariances: (N, K, D) 对角协方差矩阵形式
        """
        N, D = Y.shape
        K = self.K

        # 原始协方差截断，确保非负
        Sigma_k = safe_clip(self.covariances, self.eps)         # (K, D)
        # 样本方差截断
        theta_iter = safe_clip(theta_iter, self.eps)            # (N, D)

        # 利用广播自动扩展维度
        cov_inv = 1.0 / (Sigma_k + theta_iter[:, None, :])        # (N, K, D)                               # (N, K, D)

        # 均值差
        diff = Y[:, None, :] - self.means                       # (N, K, D)

        # 每个样本的条件均值
        mu_post = self.means + Sigma_k * cov_inv * diff         # (N, K, D)

        # 每个样本的条件协方差（对角）
        sigma_post_diag = Sigma_k - Sigma_k**2 * cov_inv        # (N, K, D)

        # 构造对角矩阵形式
        # full_covs = build_diag_cov(sigma_post_diag)             # (N, K, D, D)

        return mu_post, sigma_post_diag, sigma_post_diag

    def _logpdf(self, Y: np.ndarray, means: np.ndarray, covariances: np.ndarray) -> np.ndarray:
        """
        对角协方差下的 logpdf 计算

        参数：
            Y: (N, D) 输入样本
            means: (N, K, D) 条件均值（后验均值）
            covariances: (N, K, D) 条件协方差（对角）

        返回：
            logpdf: (N, K) 每个样本在每个分量下的 log-likelihood
        """
        cov_diag = safe_clip(covariances, self.eps)             # (N, K, D)
        diff = Y[:, None, :] - means                            # (N, K, D)

        log_det = np.sum(np.log(cov_diag), axis=2)              # (N, K)
        quad = np.sum((diff**2) / cov_diag, axis=2)             # (N, K)

        return -0.5 * (quad + log_det + Y.shape[-1] * np.log(2 * np.pi))

    def _compute_weighted_logpdf(self, Y: np.ndarray, mu_post: np.ndarray, covariances: np.ndarray) -> np.ndarray:
        """
        计算 logpdf + log(weights)，用于 SoftEM 的 gamma 计算

        参数：
            Y: (N, D)
            mu_post: (N, K, D)
            covariances: (N, K, D)

        返回：
            weighted_logpdf: (N, K)
        """
        logpdf = self._logpdf(Y, mu_post, covariances)          # (N, K)
        log_weights = np.log(safe_clip(self.weights, self.eps))  # (K,)
        return logpdf + log_weights[None, :]                             # (N, K)


