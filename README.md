# 🧠 softem

**softem** 是一个用于在线推理的轻量级推理器，专为需要在流式数据中进行概率建模的场景而设计。  
它基于历史数据训练得到的高斯混合模型参数（如 `means`、`weights`、`covariances` 和 `covariance_type`），  
通过 SoftEM 迭代方法对在线数据进行高效推理。背后基于贝叶斯推断的参数估计方法，具有较强的解释性。

---

## 📌 项目简介

在许多实际应用中，历史数据已被用于训练 `sklearn.GaussianMixture` 或 `BayesianGaussianMixture` 模型。  
**softem** 利用这些模型的参数作为初始化条件，构建一个在线推理器，能够对新到达的数据进行快速、稳定的概率推理。

核心特性包括：

- ✅ 支持从已有 GMM/BayesianGMM 模型中提取参数初始化
- 🔁 使用 SoftEM 方法进行迭代推理，适用于在线数据流
- ⚡ 高效、轻量，适合嵌入式或实时系统
- 同时也支持批量化样本

---

## 📦 安装
因还需要进行迭代和性能优化，因此暂时不进行pip的打包。
你可以通过以下方式使用 softem：

bash
```
git clone https://github.com/SJTUwanglei/softem.git
cd softem
```

---

## 🛠️ demo
examples/factory_deno.py 中设置了四种类型的工厂模式下的调用demo的一些展示。
由于代码还未优化完全，尤其是速度性能方面，因此暂时未进行 pip对应的包的封装

---

## 🚧 TODO & Roadmap

### 🛠️ 待改进
- 推理代码的numpy和numba的串行优化，或C++的局部封装
- 代码冗余清理
  - 目前为了理解维度保持较多注释
  - utils.py 保留了一些之前版本的功能函数
- 文档docs

### ✨ 计划新增功能
- 算法：增加 full 类型下Low Rank GMM
- 业务：新开辟在线推理识别的业务模块
  - 若功能复杂可能在新仓库
  - 包含业务部署
- 算法pip封装

### 🐞 已知问题
- 存在某个事实不合理的BUG但暂时很难避免，invalid value: 计算协方差矩阵的sigma 等中间数值时为负数。
  - 是因为tied、full等类型本身假设情况下，硬截断导致模型参数后续计算不稳定。
  - 即使使用对角 eps 软正则代替硬截断，目前部分仍旧 invalid value
  - 暂时解决方案只能在计算出的 sigma_post_diag 进行safe_clip截断





