import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from gmm_softem.factory import create_soft_em
from typing import List, Tuple, Dict, Any, Optional, Literal
import matplotlib.pyplot as plt
from copy import deepcopy


RANDOM_SEED = 42


def generate_sensor_data_with_anomalies(n_samples: int = 1000,
                                        n_features: int = 10,
                                        n_components: int = 3,
                                        anomaly_ratio: float = 0.5,
                                        random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成模拟传感器数据，包括正常数据和混合测试数据（含异常）

    参数:
        n_samples: 正常样本数量（训练用）
        n_features: 传感器测点数（特征维度）
        n_components: 正常数据的 GMM 组件数
        anomaly_ratio: 测试数据中异常样本的比例（0~1）
        random_state: 随机种子

    返回:
        X_all: 所有数据 (2 * n_samples, n_features)
        labels: 标签数组 (2 * n_samples,) —— 0 表示正常，1 表示异常（仅测试数据部分）
    """
    np.random.seed(random_state)
    X_normal = np.zeros((n_samples, n_features))
    samples_per_component = n_samples // n_components

    # === 正常数据生成 ===
    for i in range(n_components):
        mean = np.random.uniform(30, 80, size=n_features)
        boost_mask = np.random.rand(n_features) < 0.2
        mean[boost_mask] += np.random.uniform(100, 300, size=int(np.sum(boost_mask)))

        cov_factor = np.random.randn(n_features, n_features)
        cov = cov_factor @ cov_factor.T + np.eye(n_features) * 20

        start = i * samples_per_component
        end = (i + 1) * samples_per_component if i < n_components - 1 else n_samples
        X_normal[start:end] = np.random.multivariate_normal(mean, cov, end - start)

    # === 测试数据生成 ===
    n_test = n_samples
    n_anomaly = int(n_test * anomaly_ratio)
    n_test_normal = n_test - n_anomaly

    # 测试数据中的正常部分（复制正常生成逻辑）
    X_test_normal = np.zeros((n_test_normal, n_features))
    for i in range(n_components):
        mean = np.random.uniform(30, 80, size=n_features)
        boost_mask = np.random.rand(n_features) < 0.2
        mean[boost_mask] += np.random.uniform(100, 300, size=int(np.sum(boost_mask)))

        cov_factor = np.random.randn(n_features, n_features)
        cov = cov_factor @ cov_factor.T + np.eye(n_features) * 20

        start = i * (n_test_normal // n_components)
        end = (i + 1) * (n_test_normal // n_components) if i < n_components - 1 else n_test_normal
        X_test_normal[start:end] = np.random.multivariate_normal(mean, cov, end - start)

    # 测试数据中的异常部分（偏离正常均值）
    anomaly_mean = np.random.uniform(200, 500, size=n_features)
    anomaly_cov = np.eye(n_features) * 100
    X_test_anomaly = np.random.multivariate_normal(anomaly_mean, anomaly_cov, n_anomaly)

    # 合并测试数据（前一半正常，后一半异常）
    X_test = np.vstack([X_test_normal, X_test_anomaly])
    labels = np.array([0] * n_test_normal + [1] * n_anomaly)

    # 合并所有数据
    X_all = np.vstack([X_normal, X_test])

    return X_all, labels



def demo_soft_em_factory(cov_type: Literal["diag", "full", "tied", "spherical"],
                         X: np.ndarray,
                         n_components: int = 3,
                         max_iter: int = 30,
                         std_scaled: bool = True,
                         verbose: bool = False) -> Optional[Tuple[Any, np.ndarray, np.ndarray]]:
    """
    使用工厂模式演示SoftEM模型
    
    参数:
        cov_type: 协方差类型 ("diag", "full", "tied", "spherical")
        X: 输入数据 (n_samples, n_features)
        n_components: 组件数量
        max_iter: 最大迭代次数
        verbose: 是否输出详细信息
    
    返回:
        (soft_em_model, theta_result, x_expectation) 或 None（如果出错）
    """
    print(f"\n{'='*50}")
    print(f"演示协方差类型: {cov_type.upper()}")
    print(f"{'='*50}")
    
    # 根据协方差类型训练对应的sklearn GMM模型
    try:
        # scaler完仍然需要根据数据特性选择协方差类型
        # 即使标准化后，仍需考虑：
        # - 特征间是否相关（决定是否用full）
        # - 各组件是否应共享协方差（tied）
        # - 是否所有特征方差相同（spherical）

        gmm: GaussianMixture = GaussianMixture(n_components=n_components, 
                                               covariance_type=cov_type,
                                               random_state=RANDOM_SEED)

        X_train = X[:X.shape[0] // 2]
        X_test = X[X.shape[0] // 2:]
        scaler: StandardScaler
        if std_scaled:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
        
        gmm.fit(X_train)
        
        print(f"✓ sklearn GMM {cov_type} 模型训练完成")
        print(f"  - 均值形状: {np.asarray(gmm.means_).shape}")
        print(f"  - 协方差形状: {np.asarray(gmm.covariances_).shape}")
        print(f"  - 权重形状: {np.asarray(gmm.weights_).shape}")
    except Exception as e:
        print(f"✗ sklearn GMM训练失败: {e}")
        return None
    
    # 使用工厂模式创建SoftEM模型
    try:
        soft_em = create_soft_em(
            cov_type=cov_type,
            means=np.asarray(gmm.means_),
            covariances=np.asarray(gmm.covariances_),
            weights=np.asarray(gmm.weights_),
            max_iter=max_iter,
            tol=1e-2,
            threshold=1,
            verbose=verbose
        )
        print(f"✓ 使用工厂模式创建 SoftEM {cov_type} 模型成功")
        print(f"  - 模型类型: {type(soft_em).__name__}")
    except Exception as e:
        print(f"✗ 工厂模式创建模型失败: {e}")
        return None

    # 准备测试数据和参数：X的后一半是测试数据
    n_test: int = 100
    assert n_test < X.shape[0]//2 and X.shape[0] % n_test == 0
    idx = 0
    test_samples: np.ndarray = X_test[n_test * idx : n_test * (idx+1)]
    if std_scaled:
        test_samples = scaler.transform(test_samples)  # 取测试里的100个样本作为测试，也进行归一化
    theta_iter: np.ndarray = np.abs(np.random.randn(n_test, X.shape[1]))  # 初始化theta参数
    
    # 使用SoftEM计算条件协方差项
    try:
        mu_post: np.ndarray
        sigma_post_diag: np.ndarray
        covariances: np.ndarray
        # demo测试是为了访问protect属性的函数，实际中不可取
        mu_post, sigma_post_diag, covariances = soft_em._compute_cov_terms_batch(theta_iter, test_samples)
        print(f"✓ _compute_cov_terms_batch 计算完成")
        print(f"  - 条件均值 mu_post: {mu_post.shape}")
        print(f"  - 条件协方差 sigma_post_diag: {sigma_post_diag.shape}")
        print(f"  - 协方差 covariances: {covariances.shape}")
    except Exception as e:
        print(f"✗ _compute_cov_terms_batch 计算失败: {e}")
        return None
    
    # 执行完整的EM步骤
    try:
        theta_result: np.ndarray
        x_expectation: np.ndarray
        theta_result, x_expectation = soft_em._em_theta_step(test_samples, verbose=verbose)
        if std_scaled:
            x_expectation = scaler.inverse_transform(x_expectation)
            theta_result = theta_result * np.asarray(scaler.scale_) ** 2
        print(f"✓ _em_theta_step 执行完成")
        print(f"  - theta结果: {theta_result.shape}")
        print(f"  - 期望值: {x_expectation.shape}")
        
        # 显示部分结果
        print(f"  - theta前3行:\n{theta_result[:3]}")
        print("\n")
        print(f"  - X前3行:\n{X_test[:3]}")
        print(f"  - x_expectation前3行:\n{x_expectation[:3]}")
    except Exception as e:
        print(f"✗ _em_theta_step 执行失败: {e}")
        return None

    plotted = True
    if plotted:
        plot_predictions_with_uncertainty(x_expectation, X_test[n_test * idx : n_test * (idx+1)], theta_result)
    
    return soft_em, theta_result, x_expectation


def demo_single_covariance_type(cov_type: Literal['diag', 'full', 'tied', 'spherical'] = "diag",
                                n_samples: int = 500,
                                n_features: int = 8,
                                n_components: int = 3,
                                ) -> Optional[Tuple[Any, np.ndarray, np.ndarray]]:
    """
    演示单个协方差类型的使用
    
    参数:
        cov_type: 要演示的协方差类型
        n_samples: 样本数量
        n_features: 特征维度
        n_components: 组件数量
    
    返回:
        (soft_em_model, theta_result, x_expectation) 或 None（如果出错）
    """
    print(f"演示单个协方差类型: {cov_type}")
    
    # 生成数据
    X: np.ndarray = generate_sensor_data_with_anomalies(n_samples, n_features, n_components)[0]
    print(f"生成数据形状: {X.shape}")
    
    # 运行演示
    result = demo_soft_em_factory(cov_type, X, n_components, max_iter=21, verbose=True, std_scaled=True)
    
    if result:
        soft_em, theta_result, x_expectation = result
        print(f"\n最终结果:")
        print(f"- 模型类型: {type(soft_em).__name__}")
        print(f"- Theta结果形状: {theta_result.shape}")
        print(f"- 期望值形状: {x_expectation.shape}")
        
        return result
    
    return None


def plot_predictions_with_uncertainty(X_est: np.ndarray, Y: np.ndarray, theta: np.ndarray):
    """
    为每个特征维度绘制预测值、真实值及置信区间

    参数：
        X: 预测值，形状 (N, D)
        Y: 真实值，形状 (N, D)
        theta: 方差估计，形状 (N, D)
    """
    assert X_est.shape == Y.shape == theta.shape, "X, Y, theta must have the same shape"
    N, D = X_est.shape

    fig, axes = plt.subplots(D, 1, figsize=(10, 3 * D), sharex=True)

    axes = np.atleast_1d(axes) # 保证 axes 是可迭代的列表

    for d in range(D):
        ax = axes[d]
        x_vals = np.arange(N)

        # 主曲线
        ax.plot(x_vals, Y[:, d], color='blue', label='True Value (Y)')
        ax.plot(x_vals, X_est[:, d], color='red', label='Predicted Value (X)')

        # 置信区间
        lower_bound = Y[:, d] - np.sqrt(theta[:, d])
        upper_bound = Y[:, d] + np.sqrt(theta[:, d])
        ax.plot(x_vals, lower_bound, color='blue', linestyle='--', linewidth=1)
        ax.plot(x_vals, upper_bound, color='blue', linestyle='--', linewidth=1)

        # 添加 x 轴索引数字
        #ax.set_xticks(x_vals)
        #ax.set_xticklabels([str(i) for i in x_vals], rotation=45, fontsize=8)

        ax.set_title(f"Feature Dimension {d}")
        ax.legend(loc='upper right')
        ax.grid(True)

    plt.xlabel("Sample Index")
    plt.tight_layout()
    plt.show()


def run_all_covariance_types(n_samples: int = 1000,
                             n_features: int = 10,
                             n_components: int = 3) -> Dict[str, Dict[str, Any]]:
    """
    运行所有协方差类型的演示

    参数:
        n_samples: 样本数量
        n_features: 特征维度
        n_components: 组件数量

    返回:
        包含所有结果的字典
    """
    print("开始生成传感器数据...")
    X: np.ndarray = generate_sensor_data_with_anomalies(n_samples, n_features, n_components)[0]
    print(f"数据生成完成，形状: {X.shape}")
    print(f"数据统计 - 均值: {np.mean(X):.3f}, 标准差: {np.std(X):.3f}")

    # 定义要测试的协方差类型
    cov_types: List[Literal['diag', 'full', 'tied', 'spherical']] = ["diag", "full", "tied", "spherical"]

    # 存储结果
    results: Dict[str, Dict[str, Any]] = {}

    # 对每种协方差类型运行演示
    for cov_type in cov_types:
        try:
            result = demo_soft_em_factory(cov_type, X, n_components, max_iter=20, verbose=False)
            if result:
                soft_em, theta_result, x_expectation = result
                results[cov_type] = {
                    'model': soft_em,
                    'theta': theta_result,
                    'expectation': x_expectation,
                    'theta_stats': {
                        'mean': np.mean(theta_result),
                        'std': np.std(theta_result),
                        'min': np.min(theta_result),
                        'max': np.max(theta_result)
                    }
                }
        except Exception as e:
            print(f"处理 {cov_type} 类型时出错: {e}")

    # 比较结果
    print(f"\n{'=' * 50}")
    print("结果比较")
    print(f"{'=' * 50}")

    for cov_type, result in results.items():
        stats = result['theta_stats']
        print(f"{cov_type.upper()} 模型 theta 结果统计:")
        print(f"  - 均值: {stats['mean']:.4f}")
        print(f"  - 标准差: {stats['std']:.4f}")
        print(f"  - 最小值: {stats['min']:.4f}")
        print(f"  - 最大值: {stats['max']:.4f}")
        print()

    return results


def compare_covariance_performance(results: Dict[str, Dict[str, Any]]) -> None:
    """
    比较不同协方差类型的性能
    
    参数:
        results: run_all_covariance_types的返回结果
    """
    print(f"\n{'='*60}")
    print("协方差类型性能比较")
    print(f"{'='*60}")
    
    cov_types = list(results.keys())
    
    # 准备数据用于比较
    means = [results[cov_type]['theta_stats']['mean'] for cov_type in cov_types]
    stds = [results[cov_type]['theta_stats']['std'] for cov_type in cov_types]
    
    print("Theta值统计比较:")
    for i, cov_type in enumerate(cov_types):
        stats = results[cov_type]['theta_stats']
        print(f"  {cov_type.upper():12}: 均值={stats['mean']:8.4f}, 标准差={stats['std']:8.4f}")
    
    # 如果matplotlib可用，绘制比较图
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 均值比较
        ax1.bar(cov_types, means, color=['blue', 'green', 'red', 'orange'])
        ax1.set_title('Theta均值比较')
        ax1.set_ylabel('均值')
        
        # 标准差比较
        ax2.bar(cov_types, stds, color=['blue', 'green', 'red', 'orange'])
        ax2.set_title('Theta标准差比较')
        ax2.set_ylabel('标准差')
        
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("注意: matplotlib未安装，跳过图表绘制")


# 使用示例
if __name__ == "__main__":
    all_showed = False
    # 演示所有协方差类型
    if all_showed:
        print("运行所有协方差类型的演示:")
        all_results = run_all_covariance_types(n_samples=800, n_features=8, n_components=3)
        
        # 性能比较
        compare_covariance_performance(all_results)
        
        print("\n" + "="*70)
    print("运行单个协方差类型的详细演示:")
    # 演示单个协方差类型（带详细输出）     diag, full, spherical, tied
    demo_single_covariance_type("tied", n_samples=2000, n_features=20, n_components=5)
