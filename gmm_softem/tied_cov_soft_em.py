from .base import BaseMatrixSoftEM
import numpy as np
from .utils import safe_clip, stable_cholesky
from typing import Tuple


class TiedMatrixSoftEM(BaseMatrixSoftEM):
    def _truncate_components(self):
        indices = self._get_truncated_indices()
        self.means = self.means[indices]
        self.weights = self.weights[indices]
        # self.covariances 是共享的，不变
        self.K = len(indices)

    def _compute_cov_terms_batch(self, theta_iter: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算条件均值和协方差（共享完整协方差结构）

        参数：
            theta_iter: (N, D)
            Y: (N, D)

        返回：
            mu_post: (N, K, D)
            sigma_post_diag: (N, K, D)
            covariances: (N, D, D) 每个样本的共享协方差矩阵（用于 logpdf）
        """
        N, D = Y.shape
        K = self.K

        # 避免硬截断，改为仅对角线的eye软正则化 for invalid negative value
        # Sigma = safe_clip(self.covariances, self.eps)           # (D, D)
        Sigma = self.covariances + self.eps * np.eye(D)
        theta_diag = np.empty((N, D, D))
        idx = np.arange(D)
        # theta_diag[:, idx, idx] = safe_clip(theta_iter, self.eps)
        theta_diag[:, idx, idx] = theta_iter
        theta_diag += self.eps * np.eye(D)[None, :, :]

        cov_sum = theta_diag + Sigma                            # (N, D, D)
        cov_inv = np.linalg.pinv(cov_sum)                       # (N, D, D)
        cov_inv = np.clip(cov_inv, -self.eps, self.eps)  # 防止爆炸 for invalid

        diff = Y[:, None, :] - self.means                       # (N, K, D)
        Sigma_diff = diff @ Sigma.T                             # (N, K, D)
        mu_post = self.means + np.einsum('nij,nkj->nki', cov_inv, Sigma_diff)

        covariances = cov_sum                                   # (N, D, D)
        cov_k = Sigma - np.einsum('nij,jk->nik', cov_inv, Sigma @ Sigma.T)  # (N, D, D)
        sigma_post_diag = np.einsum('nii->ni', cov_k)[:, None, :] * np.ones((1, K, 1))  # (N, K, D)

        sigma_post_diag = safe_clip(sigma_post_diag, self.eps)  # 又来截断，避免负值

        if np.min(Sigma) < 0 or np.min(theta_iter) < 0:
            print("Min Sigma_k:", np.min(Sigma))
            print("Min theta_iter:", np.min(theta_iter))
            print("Min cov_inv:", np.min(cov_inv))
            print("Min sigma_post_diag:", np.min(sigma_post_diag))

        return mu_post, sigma_post_diag, covariances

    def _logpdf(self, Y: np.ndarray, means: np.ndarray, covariances: np.ndarray) -> np.ndarray:
        """
        使用共享协方差矩阵计算 logpdf

        参数：
            Y: (N, D)
            means: (N, K, D)
            covariances: (N, D, D)

        返回：
            logpdf: (N, K)
        """
        N, K, D = means.shape
        logpdf = np.empty((N, K))

        chol = stable_cholesky(covariances)                     # (N, D, D)
        log_det = 2 * np.sum(np.log(np.diagonal(chol, axis1=1, axis2=2)), axis=1)  # (N,)

        for k in range(K):
            diff = Y - means[:, k, :]                           # (N, D)
            solve = np.linalg.solve(chol, diff[:, :, None])    # (N, D, 1)
            quad = np.sum(solve.squeeze(-1)**2, axis=1)         # (N,)
            logpdf[:, k] = -0.5 * (quad + log_det + D * np.log(2 * np.pi))

        return logpdf

    def _compute_weighted_logpdf(self, Y: np.ndarray, mu_post: np.ndarray, covariances: np.ndarray) -> np.ndarray:
        """
        计算 logpdf + log(weights)

        参数：
            Y: (N, D)
            mu_post: (N, K, D)
            covariances: (N, D, D)

        返回：
            weighted_logpdf: (N, K)
        """
        logpdf = self._logpdf(Y, mu_post, covariances)
        log_weights = np.log(safe_clip(self.weights, self.eps))
        return logpdf + log_weights[None, :]

