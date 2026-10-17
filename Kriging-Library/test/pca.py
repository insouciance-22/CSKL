import numpy as np
import matplotlib.pyplot as plt

# 1. 生成二维高斯分布数据
np.random.seed(0)
mean = [2, 5]
cov = [[3, 1], [1, 2]]
X = np.random.multivariate_normal(mean, cov, 100)

# 2. 数据中心化
X_mean = np.mean(X, axis=0)
X_centered = X - X_mean

# 3. 计算协方差矩阵
cov_matrix = np.cov(X_centered.T)

# 4. 特征值分解（求主成分）
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# 按特征值排序（从大到小）
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# 5. 投影到主成分方向（前1个主轴）
Z = X_centered @ eigenvectors[:, :1]

# 6. 可视化
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], alpha=0.5, label='Original Data')
plt.quiver(*X_mean, *eigenvectors[:, 0]*3, color='r', scale=1, label='PC1')
plt.quiver(*X_mean, *eigenvectors[:, 1]*3, color='g', scale=1, label='PC2')
plt.title("PCA: Principal Component Directions")
plt.xlabel("X1")
plt.ylabel("X2")
plt.axis('equal')
plt.legend()
plt.grid(True)
plt.show()
