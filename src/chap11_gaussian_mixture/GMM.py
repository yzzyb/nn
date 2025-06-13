# 导入NumPy库，用于科学计算和数值操作
import numpy as np
# 导入matplotlib.pyplot模块，用于数据可视化和绘图
import matplotlib.pyplot as plt 

# 生成混合高斯分布数据
def generate_data(n_samples=1000):
    """生成混合高斯分布数据集
    
    参数:
        n_samples: 总样本数量 (默认=1000)
    
    返回:
        X: 特征矩阵 (n_samples, 2)
        y_true: 真实标签 (n_samples,)
    """
    np.random.seed(42)  # 固定随机种子以确保结果可复现
    # 定义三个高斯分布的中心点
    mu_true = np.array([ 
        [0, 0],  # 第一个高斯分布的均值
        [5, 5],  # 第二个高斯分布的均值
        [-5, 5]  # 第三个高斯分布的均值
    ])
    
    # 定义三个高斯分布的协方差矩阵
    sigma_true = np.array([
        [[1, 0], [0, 1]],  # 第一个分布：圆形分布(各向同性)
        [[2, 0.5], [0.5, 1]],  # 第二个分布：倾斜的椭圆
        [[1, -0.5], [-0.5, 2]]  # 第三个分布：反向倾斜的椭圆
    ])
    
    # 定义每个高斯分布的混合权重(必须和为1)
    weights_true = np.array([0.3, 0.4, 0.3])
    
    # 获取混合成分的数量(这里是3)
    n_components = len(weights_true)
    
    # 生成一个合成数据集，该数据集由多个多元正态分布的样本组成
    # 计算每个分布应生成的样本数（浮点转整数可能有误差）
    samples_per_component = (weights_true * n_samples).astype(int)
    
    # 确保样本总数正确（由于浮点转换可能有误差）
    total_samples = np.sum(samples_per_component)
    if total_samples < n_samples:
        # 将缺少的样本添加到权重最大的成分中
        samples_per_component[np.argmax(weights_true)] += n_samples - total_samples
    
    # 用于存储每个高斯分布生成的数据点
    X_list = []  
    
    # 用于存储每个数据点对应的真实分布标签
    y_true = []  
    
    # 从第i个高斯分布生成样本
    for i in range(n_components): 
        #生成多元正态分布样本
        X_i = np.random.multivariate_normal(mu_true[i], sigma_true[i], samples_per_component[i])
        # 将生成的样本添加到列表
        X_list.append(X_i) 
        # 添加对应标签（0、1、2表示三个分布）
        y_true.extend([i] * samples_per_component[i]) 
    
    # 合并并打乱数据并打乱顺序（模拟无标签数据）
    # 将多个子数据集合并为一个完整数据集
    X = np.vstack(X_list)  
    # 将Python列表转换为NumPy数组
    y_true = np.array(y_true) 
    # 生成0到n_samples-1的随机排列
    shuffle_idx = np.random.permutation(n_samples) 
    # 使用相同的随机索引同时打乱特征和标签
    return X[shuffle_idx], y_true[shuffle_idx]

# 自定义logsumexp函数
def logsumexp(log_p, axis=1, keepdims=False):
    """优化后的logsumexp实现，包含数值稳定性增强和特殊case处理
    
    计算log(sum(exp(log_p)))，通过减去最大值避免数值溢出
    数学公式: log(sum(exp(log_p))) = max(log_p) + log(sum(exp(log_p - max(log_p))))
    
    参数：
    log_p: 输入的对数概率（可能为负无穷）。
    axis: 沿着哪个轴进行计算，默认为1（即按行计算）。
    keepdims: 是否保持维度，默认为False。

    返回：
    计算结果的log(sum(exp(log_p)))，返回与输入数组相同形状的结果。
    """
    log_p = np.asarray(log_p)                                               # 将对数概率列表转换为NumPy数组
    
    # 处理空输入情况
    if log_p.size == 0:                                                     # 检查输入的对数概率数组是否为空
        return np.array(-np.inf, dtype=log_p.dtype)                         # 返回与输入相同数据类型的负无穷值
    
    # 计算最大值（处理全-inf输入）
    max_val = np.max(log_p, axis=axis, keepdims=True)                       # 计算沿指定轴的最大值
    if np.all(np.isneginf(max_val)):                                        # 检查是否所有最大值都是负无穷
        return max_val.copy() if keepdims else max_val.squeeze(axis=axis)   # 根据keepdims返回适当形式
    
    # 计算修正后的指数和（处理-inf输入）
    # 安全计算指数和：先减去最大值，再计算指数
    safe_log_p = np.where(np.isneginf(log_p), -np.inf, log_p - max_val)     # 安全调整对数概率
    sum_exp = np.sum(np.exp(safe_log_p), axis = axis, keepdims = keepdims)  # 计算调整后的指数和
    
    # 计算最终结果
    result = max_val + np.log(sum_exp)
    
    # 处理全-inf输入的特殊case
    if np.any(np.isneginf(log_p)) and not np.any(np.isfinite(log_p)):       # 判断是否所有有效值都是-inf
        result = max_val.copy() if keepdims else max_val.squeeze(axis=axis) # 根据keepdims参数的值返回max_val的适当形式
    return result                                                           # 返回处理后的结果，保持与正常情况相同的接口

# 高斯混合模型类
class GaussianMixtureModel:
    """高斯混合模型(GMM)实现
    
    参数:
        n_components: int, 高斯分布数量 (默认=3)
        max_iter: int, EM算法最大迭代次数 (默认=100)
        tol: float, 收敛阈值 (默认=1e-6)
        random_state: int, 随机种子 (可选)
    """
    def __init__(self, n_components = 3, max_iter = 100, tol = 1e-6, random_state = None):
        # 初始化模型参数
        self.n_components = n_components  # 高斯分布数量
        self.max_iter = max_iter          # EM算法最大迭代次数
        self.tol = tol                    # 收敛阈值
        self.log_likelihoods = []         #存储每轮迭代的对数似然值

        # 初始化随机数生成器
        self.rng = np.random.default_rng(random_state)

    def fit(self, X):
        """使用EM算法训练模型

        EM算法流程：
        1. 初始化模型参数（混合权重π、均值μ、协方差矩阵Σ）
        2. 重复以下步骤直到收敛：
           - E步：计算每个样本属于各高斯成分的后验概率（责任度）
           - M步：基于后验概率更新模型参数
        """
        X = np.asarray(X) # 将输入数据 X 转换为 NumPy 数组格式，确保后续操作的兼容性
        n_samples, n_features = X.shape # 获取数据的样本数量和特征维度
        
        # 初始化混合系数（均匀分布）
        self.pi = np.ones(self.n_components) / self.n_components
        
        # 随机选择样本点作为初始均值
        self.mu = X[np.random.choice(n_samples, self.n_components, replace=False)]
        
        # 初始化协方差矩阵为单位矩阵
        self.sigma = np.array([np.eye(n_features) for _ in range(self.n_components)])

        log_likelihood = -np.inf  # 初始化对数似然值为负无穷
        
        # EM算法主循环：交替执行E步(期望)和M步(最大化)
        for iter in range(self.max_iter):
            # E步：计算后验概率（每个样本属于各个高斯成分的概率）
            log_prob = np.zeros((n_samples, self.n_components)) # 初始化对数概率矩阵
            
            # 对每个高斯成分，计算样本的对数概率密度
            for k in range(self.n_components):
                # 对数概率 = log(混合权重) + log(高斯概率密度)
                log_prob[:, k] = np.log(self.pi[k]) + self._log_gaussian(X, self.mu[k], self.sigma[k]) # 计算第k个高斯混合成分的对数概率密度，并存储在log_prob的第k列
            
            # 使用logsumexp计算归一化因子，确保数值稳定性
            log_prob_sum = logsumexp(log_prob, axis=1, keepdims=True)
            
            # 计算后验概率（responsibility）：gamma_{ik} = P(z_i=k|x_i)
            gamma = np.exp(log_prob - log_prob_sum)

            # M步：更新模型参数（基于后验概率）
            Nk = np.sum(gamma, axis=0) # 每个高斯成分的"有效样本数"
            
            # 更新混合权重
            self.pi = Nk / n_samples
            
            # 初始化新均值和新协方差矩阵
            new_mu = np.zeros_like(self.mu)# 创建一个与 self.mu 形状相同且全为零的数组，作为新的均值向量
            new_sigma = np.zeros_like(self.sigma)# 创建一个与 self.sigma 形状相同且全为零的数组，作为新的协方差矩阵

            # 对每个高斯成分更新参数
            for k in range(self.n_components):
                # 更新均值：加权平均
                new_mu[k] = np.sum(gamma[:, k, None] * X, axis=0) / Nk[k]

                # 更新协方差矩阵
                X_centered = X - new_mu[k]  # 中心化数据
                weighted_X = gamma[:, k, None] * X_centered  # 加权中心化数据
                
                # 使用einsum高效计算协方差矩阵
                # 等价于: new_sigma_k = (X_centered.T @ diag(gamma[:,k]) @ X_centered) / Nk[k]
                new_sigma_k = np.einsum('ni,nj->ij', X_centered, weighted_X) / Nk[k]
                
                # 正则化：添加小的对角矩阵，防止协方差矩阵奇异
                eps = 1e-6  # 正则化系数
                new_sigma_k += np.eye(n_features) * eps  # 添加单位矩阵的eps倍
                
                new_sigma[k] = new_sigma_k  # 存储更新后的协方差矩阵

            # 计算对数似然（模型对数据的拟合程度）
            current_log_likelihood = np.sum(log_prob_sum)  # 所有样本的对数似然之和
            self.log_likelihoods.append(current_log_likelihood)  # 记录当前对数似然
            
            # 检查收敛条件：如果对数似然变化小于阈值，则停止迭代
            if iter > 0 and abs(current_log_likelihood - log_likelihood) < self.tol:
                break
                
            log_likelihood = current_log_likelihood   # 更新记录的上一次迭代的对数似然值
            
            # 更新模型参数
            self.mu = new_mu
            self.sigma = new_sigma
        
        # 最终聚类结果：每个样本分配到概率最大的高斯成分
        self.labels_ = np.argmax(gamma, axis=1)
        # 基于软聚类结果确定最终的硬聚类标签
        return self

    def _log_gaussian(self, X, mu, sigma):
        """计算多维高斯分布的对数概率密度
        
        参数:
            X: 输入数据点/样本集，形状为(n_samples, n_features)
            mu: 高斯分布的均值向量，形状为(n_features,)
            sigma: 高斯分布的协方差矩阵，形状为(n_features, n_features)
            
        返回:
            log_prob: 每个样本的对数概率密度，形状为(n_samples,)
        """
        # 获取特征维度数（协方差矩阵的维度）
        n_features = mu.shape[0]

        # 数据归一化：将数据减去均值，得到中心化数据
        # 高斯分布公式中的(x-μ)项
        X_centered = X - mu  # 形状保持(n_samples, n_features)

        # 计算协方差矩阵的行列式符号和对数值
        # sign: 行列式的符号（正负）
        # logdet: 行列式的自然对数值
        sign, logdet = np.linalg.slogdet(sigma)  # 数值稳定的行列式计算方法

        # 处理协方差矩阵可能奇异（不可逆）的情况
        if sign <= 0:  # 行列式非正（理论上协方差矩阵应是正定的）
            # 添加一个小的对角扰动项（单位矩阵乘以1e-6）
            # 确保矩阵可逆且正定，提高数值稳定性
            sigma += np.eye(n_features) * 1e-6  # 正则化处理
            
            # 重新计算调整后的协方差矩阵的行列式
            sign, logdet = np.linalg.slogdet(sigma)

            # 计算协方差矩阵的逆
            inv = np.linalg.inv(sigma)
            
            # 计算二次型：(x-μ)^T·Σ^(-1)·(x-μ)
            # 使用einsum高效计算多个样本的二次型
            exponent = -0.5 * np.einsum('...i,...i->...', X_centered @ inv, X_centered)

            # 返回对数概率密度
            # 公式：log_p(x) = -0.5*D*log(2π) - 0.5*log|Σ| - 0.5*(x-μ)^T·Σ^(-1)·(x-μ)
            return -0.5 * n_features * np.log(2 * np.pi) - 0.5 * logdet + exponent
        else:
            # 处理非奇异协方差矩阵
            inv = np.linalg.inv(sigma) #计算协方差矩阵的逆
            exponent = -0.5 * np.einsum('...i,...i->...', X_centered @ inv, X_centered) #计算指数部分（二次型）
            return -0.5 * n_features * np.log(2 * np.pi) - 0.5 * logdet + exponent #组合对数概率密度
        
    def plot_convergence(self):
        """可视化对数似然的收敛过程"""
        # 检查是否有对数似然值记录
        if not self.log_likelihoods:
            raise ValueError("请先调用fit方法训练模型")

        # 创建一个图形窗口，设置大小为10x6英寸
        plt.figure(figsize=(10, 6))
        # 绘制对数似然值随迭代次数的变化曲线
        # 使用蓝色实线绘制，范围从1到len(self.log_likelihoods)
        plt.plot(range(1, len(self.log_likelihoods) + 1), self.log_likelihoods, 'b-')
        # 设置x轴的标签为“迭代次数”
        plt.xlabel('迭代次数')
        # 设置y轴的标签为“对数似然值”
        plt.ylabel('对数似然值')
        # 设置图表的标题为“EM算法收敛曲线”
        plt.title('EM算法收敛曲线')
        # 启用网格线，增强可读性
        plt.grid(True, alpha=0.5) 
        plt.show()

# 主程序
if __name__ == "__main__":
    # 1. 生成合成数据
    print("生成混合高斯分布数据...")
    # 调用generate_data函数生成样本数据：
    X, y_true = generate_data(n_samples=1000)
    print(f"生成数据形状: {X.shape}, 标签形状: {y_true.shape}")
    
    # 训练GMM模型
    gmm = GaussianMixtureModel(n_components=3)
    gmm.fit(X)
    y_pred = gmm.labels_
     #
     
    # 可视化结果
    plt.figure(figsize=(12, 5))#创建图形和子图
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', s=10)#绘制散点图，x 轴为 X 的第一列（Feature 1），y 轴为 X 的第二列（Feature 2）；点颜色由 y_true 决定（真实聚类标签）
    # 使用 viridis 颜色映射；s=10 设置点的大小为 10
    plt.title("True Clusters")
    # 注意：此处重复设置标题是为了确保在某些环境中标题能够正确显示
    plt.title("True Clusters")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', s=10)#绘制散点图，x 轴和 y 轴与第一个子图相同。点颜色由 y_pred 决定（GMM 预测的聚类标签），
    # 使用相同的 viridis 颜色映射。点大小同样为 10。
    plt.title("GMM Predicted Clusters")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()#显示创建的图形窗口
    #打印信息
    print("生成混合高斯分布数据...")
    print("生成混合高斯分布数据...")
    print("生成混合高斯分布数据...")




