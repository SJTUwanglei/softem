# soft_em/factory.py
from typing import cast, Literal, Dict, Any
import numpy as np
from gmm_softem import (
    DiagMatrixSoftEM,
    FullMatrixSoftEM,
    TiedMatrixSoftEM,
    SphericalMatrixSoftEM,
)


def create_soft_em(
    cov_type: Literal['diag', 'full', 'tied', 'spherical'],
    means: np.ndarray,
    covariances: np.ndarray,
    weights: np.ndarray,
    threshold: float = 0.95,
    verbose: bool = False,
    **kwargs: Any
):
    """
    创建 SoftEM 推理器实例。

    参数：
        cov_type (Literal['diag', 'full', 'tied', 'spherical']): 协方差类型
            - 'diag': 对角协方差矩阵，每个分量有自己独立的对角协方差矩阵
            - 'full': 全协方差矩阵，每个分量有自己独立的完整协方差矩阵
            - 'tied': 共享协方差矩阵，所有分量共享同一个完整协方差矩阵
            - 'spherical': 球面协方差矩阵，每个分量协方差矩阵是标量乘以单位矩阵
        means (np.ndarray): 均值数组，形状为 (K, D)
            - K: 分量数量
            - D: 数据维度（特征数）
        covariances (np.ndarray): 协方差结构，形状依赖于 cov_type
            - 'diag': (K, D) - 每个分量的对角元素
            - 'full': (K, D, D) - 每个分量的完整协方差矩阵
            - 'tied': (D, D) - 所有分量共享的协方差矩阵
            - 'spherical': (K,) - 每个分量的标量协方差值
        weights (np.ndarray): 分量权重，形状为 (K,)
            - 每个分量的先验概率，所有权重之和为1
        **kwargs: 其他传递给构造函数的参数，如:
            - max_iter (int): 最大迭代次数，默认值根据具体实现而定
            - tol (float): 收敛容差，默认值根据具体实现而定
            - verbose (bool): 是否输出详细信息，默认False

    返回：
        BaseMatrixSoftEM 的子类实例:
            - DiagMatrixSoftEM: 当 cov_type 为 'diag'
            - FullMatrixSoftEM: 当 cov_type 为 'full'
            - TiedMatrixSoftEM: 当 cov_type 为 'tied'
            - SphericalMatrixSoftEM: 当 cov_type 为 'spherical'

    异常：
        ValueError: 当 cov_type 不是支持的类型时抛出
    """

    cov_type = cast(Literal['diag', 'full', 'tied', 'spherical'], cov_type.lower())

    if cov_type == "diag":
        return DiagMatrixSoftEM(means, covariances, weights, threshold=threshold, verbose=verbose, **kwargs)
    elif cov_type == "full":
        return FullMatrixSoftEM(means, covariances, weights, threshold=threshold, verbose=verbose, **kwargs)
    elif cov_type == "tied":
        return TiedMatrixSoftEM(means, covariances, weights, threshold=threshold, verbose=verbose, **kwargs)
    elif cov_type == "spherical":
        return SphericalMatrixSoftEM(means, covariances, weights, threshold=threshold, verbose=verbose, **kwargs)
    else:
        raise ValueError(f"Unknown covariance type: '{cov_type}'. "
                         f"Supported types are: 'diag', 'full', 'tied', 'spherical'")




