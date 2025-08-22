import numpy as np
import sys
from typing import List

# 检查 Python 版本
assert sys.version_info >= (3, 7), f"需要 Python >= 3.7，但当前版本为 {sys.version_info.major}.{sys.version_info.minor}"

# 检查 NumPy 版本
required_numpy_version = (1, 20)
current_numpy_version = tuple(map(int, np.__version__.split('.')[:2]))
assert current_numpy_version >= required_numpy_version, f"因为batch cholesky分解需要 NumPy >= 1.20，但当前版本为 {np.__version__}"


def stable_inverse(matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    稳定求逆：奇异矩阵使用伪逆

    参数：
        matrix: 输入矩阵
        eps: 稳定性参数

    返回：
        矩阵的逆或伪逆
    """
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(matrix + eps * np.eye(matrix.shape[-1]))


def stable_cholesky(matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    稳定 Cholesky 分解

    参数：
        matrix: 输入矩阵
        eps: 稳定性参数

    返回：
        Cholesky 分解结果
    """
    """
    稳定 Cholesky 分解
    前提是你的 NumPy 版本支持 batch Cholesky 分解（即对形状为 (N, D, D) 的数组进行逐个矩阵分解）。
    这个功能是在 NumPy 1.20 及之后版本中引入的。如果你输入的matrix是三维的，当然二维是肯定支持的。
    对应的，此版本支持的 Python 版本为 3.7-3.9，已取消对 Python 3.6 的支持。
    """
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        return np.linalg.cholesky(matrix + eps * np.eye(matrix.shape[-1]))


# 拓展版本，暂时不需要for full cov    @njit and fast version
import numba
from numba import njit
@njit
def stable_cholesky_numba(matrix_flat, eps_init=1e-8, max_attempts=5):
    """
    底层使用的是线程池加速，这会导致系统资源紧张，注意！
    """
    N, D, _ = matrix_flat.shape
    chol_flat = np.empty_like(matrix_flat)
    fix_mask = np.zeros(N, dtype=np.bool_)

    I = np.eye(D)

    for i in range(N):
        mat = matrix_flat[i]
        eps = eps_init
        success = False

        for _ in range(max_attempts):
            try:
                chol = np.linalg.cholesky(mat)
                chol_flat[i] = chol
                success = True
                break
            except np.linalg.LinAlgError:
                mat = mat + eps * I
                eps *= 10

        if not success:
            chol_flat[i] = np.full((D, D), np.nan)
            fix_mask[i] = True

    return chol_flat, fix_mask


def stable_cholesky_fast(matrix, eps=1e-8, max_attempts=5, return_mask=False):
    """
    matrix: (N, K, D, D) 以及更高维度
    包装 numba 加速的 Cholesky 分解，支持任意批量维度。
    """
    shape = matrix.shape
    D = shape[-1]
    batch_shape = shape[:-2]
    matrix_flat = matrix.reshape(-1, D, D)

    chol_flat, fix_mask = stable_cholesky_numba(matrix_flat, eps, max_attempts)
    chol = chol_flat.reshape(*batch_shape, D, D)
    fix_mask = fix_mask.reshape(*batch_shape)

    if return_mask:
        return chol, fix_mask
    else:
        return chol


def stable_softmax(x: np.ndarray, axis = None, keepdims: bool = True) -> np.ndarray:
    """
    数值稳定的softmax实现

    参数：
        x: 输入数组
        axis: 计算轴
        keepdims: 是否保持维度

    返回：
        softmax结果
    """
    x_max = np.max(x, axis=axis, keepdims=keepdims)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=keepdims)


def weights_cumsum_select(weights: np.ndarray, threshold: float = 0.95) -> List[int]:
    """
    从 GMM 模型中筛选累计权重达到95%的组件索引（返回原始顺序的索引）

    参数：
        weights: 训练好的 GaussianMixture 模型的weights
        threshold: 累计权重阈值，默认 0.95

    返回：
        selected_indices: list[int]，原始顺序下的组件索引
    """
    sorted_idx = np.argsort(weights)[::-1]  # 从大到小排序
    cum_weights = np.cumsum(weights[sorted_idx])

    # 找到累计权重达到阈值的位置
    cutoff = np.searchsorted(cum_weights, threshold) + 1
    top_idx_sorted = sorted_idx[:cutoff]

    # 转换为原始顺序下的索引
    selected_mask = np.zeros_like(weights, dtype=bool)
    selected_mask[top_idx_sorted] = True
    selected_indices = np.where(selected_mask)[0].tolist()
    print(f"weights cumsum reserved threshold: {threshold}, indices ratio: {len(selected_indices)}/{len(weights)}")

    return selected_indices


def safe_clip(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    确保数组中所有元素都不小于 eps

    参数：
        x: 输入数组
        eps: 最小值

    返回：
        截断后的数组
    """
    return np.clip(x, eps, np.inf)      # 代替np.maximum()


def batch_convergence_loop(
    init_state,                  # 初始状态 (N, D)
    update_fn,                   # 更新函数：接受当前状态，返回新状态
    convergence_fn,              # 收敛判断函数：接受旧状态和新状态，返回布尔数组 (N,)
    max_iter=50,                 # 最大迭代次数
    verbose=False                # 是否打印进度
):
    """
    针对批量样本进行迭代，直到所有样本收敛。

    参数：
    - init_state: 初始状态数组，形状为 (N, D)
    - update_fn: 更新函数，输入当前状态，输出新状态
    - convergence_fn: 收敛判断函数，输入旧状态和新状态，输出布尔数组
    - max_iter: 最大迭代次数
    - verbose: 是否打印每轮收敛情况

    返回：
    - final_state: 所有样本最终状态
    - converge_steps: 每个样本收敛所用的迭代次数
    """
    N = init_state.shape[0]
    state = init_state.copy()
    converge_mask = np.zeros(N, dtype=bool)
    converge_steps = np.full(N, -1)

    for t in range(max_iter):
        new_state = update_fn(state)

        # 判断哪些样本收敛
        has_converged = convergence_fn(state, new_state)
        newly_converged = has_converged & (~converge_mask)
        converge_mask |= has_converged
        converge_steps[newly_converged] = t

        # 冻结已收敛样本
        state[~converge_mask] = new_state[~converge_mask]

        if verbose:
            print(f"Iter {t}: {np.sum(converge_mask)} / {N} samples converged")

        if np.all(converge_mask):
            break

    return state, converge_steps




def safe_diag_matrix(x, eps=1e-8):
    """将向量列表转换为对角矩阵，并进行截断"""
    x = safe_clip(x, eps)
    return np.array([np.diag(row) for row in x])


def build_diag_cov(sigma_diag: np.ndarray) -> np.ndarray:
    """
    将对角协方差向量构造为对角矩阵形式

    参数：
        sigma_diag: 对角协方差向量

    返回：
        对角协方差矩阵
    """
    N, K, D = sigma_diag.shape
    covs = np.zeros((N, K, D, D))
    covs[:, :, np.arange(D), np.arange(D)] = sigma_diag
    return covs


def logpdf_diag(X: np.ndarray, means: np.ndarray, cov_diag: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    对角协方差的 logpdf 计算

    参数：
        X: 输入数据
        means: 均值
        cov_diag: 对角协方差
        eps: 稳定性参数

    返回：
        logpdf值
    """
    cov_diag = safe_clip(cov_diag, eps)
    diff = X[:, None, :] - means  # (N, K, D)
    log_det = np.sum(np.log(cov_diag), axis=2)  # (N, K)
    quad = np.sum((diff**2) / cov_diag, axis=2)  # (N, K)
    D = X.shape[1]
    return -0.5 * (quad + log_det + D * np.log(2 * np.pi))


def logpdf_full(X: np.ndarray, means: np.ndarray, covs: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    全协方差的 logpdf 计算（逐样本）

    参数：
        X: 输入数据
        means: 均值
        covs: 协方差矩阵
        eps: 稳定性参数

    返回：
        logpdf值
    """
    N, K, D = means.shape
    logpdf = np.zeros((N, K))
    for k in range(K):
        for n in range(N):
            chol = stable_cholesky(covs[n, k], eps)
            solve = np.linalg.solve(chol, X[n] - means[n, k])
            log_det = 2 * np.sum(np.log(safe_clip(np.diag(chol), eps)))
            quad = np.sum(solve**2)
            logpdf[n, k] = -0.5 * (quad + log_det + D * np.log(2 * np.pi))
    return logpdf