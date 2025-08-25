import numpy as np
from .base import BaseMatrixSoftEM
from .utils import safe_clip, stable_cholesky, stable_inverse
from typing import Tuple


class FullMatrixSoftEM(BaseMatrixSoftEM):
    def _truncate_components(self):
        indices = self._get_truncated_indices()
        self.means = self.means[indices]
        self.weights = self.weights[indices]
        self.covariances = self.covariances[indices]
        self.K = len(indices)

    def _compute_cov_terms_batch(self, theta_iter: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        向量化计算条件均值和协方差（完整协方差结构）

        参数：
            theta_iter: (N, D) 当前样本的条件方差估计
            Y: (N, D) 输入样本

        返回：
            mu_post: (N, K, D)
            sigma_post_diag: (N, K, D)
            full_covs: (N, K, D, D)
        """
        N, D = Y.shape
        K = self.K

        # Clip covariance and theta
        Sigma_k = safe_clip(self.covariances, self.eps)  # (K, D, D)
        theta_iter = safe_clip(theta_iter, self.eps)  # (N, D)

        # 2. 构造 theta 的对角矩阵（向量化，避免显式循环）
        theta_diag = np.einsum('nd,ij->nij', theta_iter, np.eye(D))  # (N, D, D)

        # 扩展维度以便广播： (N, K, D, D)
        theta_diag_exp = theta_diag[:, None, :, :]  # (N, 1, D, D)
        Sigma_k_exp = Sigma_k[None, :, :, :]  # (1, K, D, D)

        # 条件协方差总和： (N, K, D, D)
        cov_sum = theta_diag_exp + Sigma_k_exp

        # 求逆： (N, K, D, D)
        cov_inv = stable_inverse(cov_sum.reshape(-1, D, D)).reshape(N, K, D, D)

        # 均值差： (N, K, D)
        diff = Y[:, None, :] - self.means  # (N, K, D)

        # 计算 mu_post 向量化： (N, K, D)
        Sigma_k_diff = np.einsum('kij,nkj->nki', Sigma_k, diff)  # (N, K, D)
        mu_post = self.means + np.einsum('nkij,nkj->nki', cov_inv, Sigma_k_diff)

        # 计算 full_covs 向量化： (N, K, D, D)
        Sigma_k_exp2 = Sigma_k[None, :, :, :]  # (1, K, D, D)
        full_covs = Sigma_k_exp2 - np.einsum('kij,nkjl->nkil', Sigma_k,
                                             np.einsum('nkij,kjl->nkil', cov_inv, Sigma_k))

        # 提取对角元素： (N, K, D)
        # sigma_post_diag = np.einsum('nkdd->nkd', full_covs)
        sigma_post_diag = np.diagonal(full_covs, axis1=2, axis2=3)  # (N, K, D)  仅仅这一步用np.diagonal代替einsum速度更快

        return mu_post, sigma_post_diag, full_covs

    def _logpdf(self, Y: np.ndarray, means: np.ndarray, covariances: np.ndarray) -> np.ndarray:
        """
        使用完整协方差矩阵计算 logpdf（完全向量化）

        参数：
            Y: (N, D)
            means: (N, K, D)
            covariances: (N, K, D, D)

        返回：
            logpdf: (N, K)
        """
        N, K, D = means.shape

        # 差值 (N, K, D)
        diff = Y[:, None, :] - means

        # Cholesky 分解 (N, K, D, D)
        chol = stable_cholesky(covariances)

        # 解线性系统：solve L x = diff.T
        # 转换为 (N*K, D, D) 和 (N*K, D, 1)，支持3维batch但不支持4维
        chol_reshaped = chol.reshape(-1, D, D)
        diff_reshaped = diff.reshape(-1, D, 1)
        solve = np.linalg.solve(chol_reshaped, diff_reshaped).reshape(N, K, D, 1)

        # Mahalanobis 距离 (N, K)
        quad = np.sum(solve.squeeze(-1) ** 2, axis=-1)

        # 对数行列式 (N, K)
        log_diag = np.log(np.diagonal(chol, axis1=2, axis2=3))  # (N, K, D)
        log_det = 2 * np.sum(log_diag, axis=-1)  # (N, K)

        # 最终 logpdf
        logpdf = -0.5 * (quad + log_det + D * np.log(2 * np.pi))  # (N, K)
        return logpdf


    def _iterK_logpdf(self, Y: np.ndarray, means: np.ndarray, covariances: np.ndarray) -> np.ndarray:
        """
        使用完整协方差矩阵计算 logpdf

        参数：
            Y: (N, D)
            means: (N, K, D)
            covariances: (N, K, D, D)

        返回：
            logpdf: (N, K)
        """
        N, K, D = means.shape
        logpdf = np.empty((N, K))

        # 此处针对组件的for循环
        for k in range(K):
            cov_k = covariances[:, k, :, :]  # (N, D, D)
            mean_k = means[:, k, :]  # (N, D)
            diff = Y - mean_k  # (N, D)

            chol = stable_cholesky(cov_k)  # (N, D, D)
            log_det = 2 * np.sum(np.log(np.diagonal(chol, axis1=1, axis2=2)), axis=1)  # (N,)
            solve = np.linalg.solve(chol, diff[:, :, None])  # (N, D, 1)
            quad = np.sum(solve.squeeze(-1) ** 2, axis=1)  # (N,)

            logpdf[:, k] = -0.5 * (quad + log_det + D * np.log(2 * np.pi))

        return logpdf

    def _compute_weighted_logpdf(self, Y: np.ndarray, mu_post: np.ndarray, covariances: np.ndarray) -> np.ndarray:
        """
        计算 logpdf + log(weights)，用于 SoftEM 的 gamma 计算

        参数：
            Y: (N, D)
            mu_post: (N, K, D)
            covariances: (N, K, D, D)

        返回：
            weighted_logpdf: (N, K)
        """
        logpdf = self._logpdf(Y, mu_post, covariances)
        log_weights = np.log(safe_clip(self.weights, self.eps))
        return logpdf + log_weights[None:]
