import numpy as np
from .base import BaseMatrixSoftEM
from .utils import safe_clip
from typing import Tuple


class SphericalMatrixSoftEM(BaseMatrixSoftEM):
    def _truncate_components(self):
        indices = self._get_truncated_indices()
        self.means = self.means[indices]
        self.weights = self.weights[indices]
        self.covariances = self.covariances[indices]
        self.K = len(indices)

    def _compute_cov_terms_batch(self, theta_iter: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算条件均值和协方差（球形结构）

        参数：
            theta_iter: (N, D) 每个样本的条件方差估计
            Y: (N, D) 输入样本

        返回：
            mu_post: (N, K, D) 条件均值
            sigma_post_diag: (N, K, D) 条件协方差的对角元素（广播标量）
            covariances: (N, D) 每个样本的球形协方差标量（用于 logpdf）
        """
        N, D = Y.shape
        K = self.K

        sigma2_k = safe_clip(self.covariances, self.eps)        # (K,)
        theta_iter = safe_clip(theta_iter, self.eps)            # (N, D)

        # 每个样本的平均方差作为球形协方差
        theta_scalar = np.mean(theta_iter, axis=1)              # (N,)
        cov_sum = sigma2_k[None, :] + theta_scalar[:, None]     # (N, K)
        cov_inv = 1.0 / cov_sum                                 # (N, K)

        diff = Y[:, None, :] - self.means                       # (N, K, D)
        mu_post = self.means + sigma2_k[None, :, None] * cov_inv[:, :, None] * diff  # (N, K, D)

        sigma_scalar = sigma2_k[None, :] - sigma2_k[None, :]**2 * cov_inv  # (N, K)
        sigma_post_diag = np.repeat(sigma_scalar[:, :, None], D, axis=2)   # (N, K, D)

        # 输出每个样本的球形协方差标量（用于 logpdf）
        covariances = theta_scalar                              # (N,)

        return mu_post, sigma_post_diag, covariances

    def _logpdf(self, Y: np.ndarray, means: np.ndarray, covariances: np.ndarray) -> np.ndarray:
        """
        使用球形协方差结构计算 logpdf

        参数：
            Y: (N, D)
            means: (N, K, D)
            covariances: (N,) 每个样本的球形协方差标量

        返回：
            logpdf: (N, K)
        """
        N, K, D = means.shape
        var = safe_clip(covariances, self.eps)                  # (N,)
        diff = Y[:, None, :] - means                            # (N, K, D)

        quad = np.sum(diff**2, axis=2) / var[:, None]           # (N, K)
        log_det = D * np.log(var[:, None])                      # (N, K)

        return -0.5 * (quad + log_det + D * np.log(2 * np.pi))

    def _compute_weighted_logpdf(self, Y: np.ndarray, mu_post: np.ndarray, covariances: np.ndarray) -> np.ndarray:
        """
        计算 logpdf + log(weights)，用于 SoftEM 的 gamma 计算

        参数：
            Y: (N, D)
            mu_post: (N, K, D)
            covariances: (N,) 球形协方差标量

        返回：
            weighted_logpdf: (N, K)
        """
        logpdf = self._logpdf(Y, mu_post, covariances)
        log_weights = np.log(safe_clip(self.weights, self.eps))  # (K,)
        return logpdf + log_weights[None, :]

