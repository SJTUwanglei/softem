from .base import BaseMatrixSoftEM
from .diag_cov_soft_em import DiagMatrixSoftEM
from .full_cov_soft_em import FullMatrixSoftEM
from .tied_cov_soft_em import TiedMatrixSoftEM
from .spherical_cov_soft_em import SphericalMatrixSoftEM

__all__ = [
    "BaseMatrixSoftEM",
    "DiagMatrixSoftEM",
    "FullMatrixSoftEM",
    "TiedMatrixSoftEM",
    "SphericalMatrixSoftEM",
]


"""
# 有 __init__.py 的统一导出
from soft_em import DiagonalCovSoftEM

# 没有统一导出时
from gmm_softem.diag_cov_soft_em import DiagMatrixSoftEM

✅ 保留 __init__.py 的统一导出，让你的包更易用。
✅ 使用工厂函数，让你的系统更灵活、可配置。
✅ 两者结合，是成熟 Python 包设计的常见模式。

"""