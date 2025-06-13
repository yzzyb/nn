#!/usr/bin/env python
# coding: utf-8
# ## 准备数据
# In[1]:

# 导入 NumPy 数值计算库，用于高效处理多维数组和矩阵运算
import numpy as np
# 导入 TensorFlow 深度学习框架
import tensorflow as tf
# 从 tqdm 库中导入 tqdm 函数
from tqdm import tqdm
# 从 TensorFlow 中导入 Keras 高级 API
from tensorflow import keras
# 从 Keras 中导入常用模块
from tensorflow.keras import layers, optimizers, datasets

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}


# 定义了一个函数 mnist_dataset()，用于加载并预处理 MNIST 数据集
def mnist_dataset():
    (x, y), (x_test, y_test) = datasets.mnist.load_data()
    # normalize 归一化：将像素值从 [0, 255] 缩放到 [0, 1] 范围内
    x = x / 255.0
    x_test = x_test / 255.0

    # 返回处理后的训练集和测试集
    # 返回格式：(训练图像数组, 训练标签数组), (测试图像数组, 测试标签数组)
    return (x, y), (x_test, y_test)


# ## Demo numpy based auto differentiation
# In[3]:


# 定义矩阵乘法层。矩阵乘法层是神经网络的“线性变换引擎”，通过权重矩阵实现高维空间的映射，为非线性计算提供基础。


class Matmul:
    def __init__(self):      
# 初始化内存字典，用于存储前向传播中的变量以便反向传播使用
        self.mem = {}

    def forward(self, x, W):
        # 前向传播：执行矩阵乘法，计算 h = x @ W
        h = np.matmul(x, W)
        # 缓存输入 x 和 权重 W，以便在反向传播中计算梯度
        self.mem = {'x': x, 'W': W}
        return h

    def backward(self, grad_y):
        '''
        x: shape(N, d)
        w: shape(d, d')
        grad_y: shape(N, d')
        '''
        # 反向传播计算 x 和 W 的梯度
        x = self.mem['x']
        W = self.mem['W']

        '''计算矩阵乘法的对应的梯度'''
        # 计算输入x的梯度：将输出梯度grad_y通过权重矩阵W的转置进行反向传播
        grad_x = np.matmul(grad_y, W.T)
        # 执行矩形乘法运算，计算梯度
        grad_W = np.matmul(x.T, grad_y)
        
        return grad_x, grad_W


# 定义 ReLU 激活层
class Relu:
    def __init__(self):
        self.mem = {}
        # 初始化记忆字典，用于存储前向传播的输入

    def forward(self, x):
        # 保存输入 x，供反向传播使用
        self.mem['x'] = x
        return np.where(x > 0, x, np.zeros_like(x))
    # ReLU激活函数：x>0时输出x，否则输出0

    def backward(self, grad_y):
        '''
        grad_y: same shape as x
        '''
        ####################
        '''计算 relu 激活函数对应的梯度'''
        x = self.mem['x']
        # ReLU的梯度是1（x>0）或0（x<=0）
        grad_x = grad_y * (x > 0)
        ####################
        return grad_x


# 定义 Softmax 层（输出概率）
class Softmax:
    '''
    softmax over last dimention
    '''
    def __init__(self):
        # 初始化类实例的基础参数和状态容器
        self.epsilon = 1e-12
        # 初始化一个空字典 mem，用于存储中间计算结果（如缓存），避免重复计算。
        self.mem = {}

    def forward(self, x):
        '''
        x: shape(N, c)
        '''
        # 对输入数据应用指数函数，确保所有值为正
        x_exp = np.exp(x)
        # 计算每个样本的归一化分母（分区函数）
        partition = np.sum(x_exp, axis=1, keepdims=True)
        # 计算 softmax 输出：指数值 / 分区函数
        # 添加 epsilon 防止除零错误（数值稳定性）
        out = x_exp / (partition + self.epsilon)

        # 将计算结果存入内存字典，用于反向传播
        self.mem['out'] = out
        self.mem['x_exp'] = x_exp
        return out

    def backward(self, grad_y):
        '''
        grad_y: same shape as x
        '''
        s = self.mem['out']
        sisj = np.matmul(np.expand_dims(s, axis=2), np.expand_dims(s, axis=1))  # (N, c, c)
        # 对 grad_y 进行维度扩展
        # 假设 grad_y 是一个形状为 (N, c) 的梯度张量
        # np.expand_dims(grad_y, axis=1) 将其形状变为 (N, 1, c)
        g_y_exp = np.expand_dims(grad_y, axis=1)
        # (N, 1, c)
        tmp = np.matmul(g_y_exp, sisj) # 计算矩阵乘法结果
        tmp = np.squeeze(tmp, axis=1)  # 去掉结果矩阵的单维度条目
        tmp = -tmp + grad_y * s # 对变量 tmp 进行更新操作
        return tmp


# 定义 Log 层（计算 log softmax ，用于交叉熵）
class Log:
    '''
    对最后一个维度执行对数运算
    用于对数似然计算，通常与Softmax结合使用
    实现数值稳定的对数计算，并保存中间结果用于反向传播
    '''
    def __init__(self):
        # 设置一个极小值epsilon防止对数运算中出现log(0)的情况
        self.epsilon = 1e-12  
        # 用于存储前向传播的中间结果，供反向传播使用
        self.mem = {}  

    def forward(self, x):
        '''
        前向传播：计算输入的对数值
        :param x: 输入数据，shape(N, c) 
                   N是batch大小，c是类别数/特征维度
        :return: log(x + epsilon)，保持数值稳定性
        '''
        # 计算对数，加上epsilon避免x=0时出现NaN
        out = np.log(x + self.epsilon)  

        # 保存输入x用于反向传播计算
        self.mem['x'] = x  
        return out

    def backward(self, grad_y):
        '''
        反向传播：计算梯度
        :param grad_y: 上游传来的梯度，shape与forward输入x相同
        :return: 当前层的梯度 = (1/(x + epsilon)) * grad_y
        '''
        # 从内存中取出前向传播保存的输入x
        x = self.mem['x']  

        # 计算当前层梯度：d(log(x))/dx = 1/x
        # 乘以来自上游的梯度grad_y（链式法则）
        return 1. / (x + self.epsilon) * grad_y  


# ## Gradient check

# In[5]:

# import tensorflow as tf

# x = np.random.normal(size=[5, 6])
# W = np.random.normal(size=[6, 4])
# aa = Matmul()
# out = aa.forward(x, W)  # shape(5, 4)
# grad = aa.backward(np.ones_like(out))
# print(grad)

# with tf.GradientTape() as tape:
#     x, W = tf.constant(x), tf.constant(W)
#     tape.watch(x)
#     y = tf.matmul(x, W)
#     loss = tf.reduce_sum(y)
#     grads = tape.gradient(loss, x)
#     print(grads)

# import tensorflow as tf

# x = np.random.normal(size=[5, 6])
# aa = Relu()
# out = aa.forward(x)  # shape(5, 4)
# grad = aa.backward(np.ones_like(out))
# print(grad)

# with tf.GradientTape() as tape:
#     x = tf.constant(x)
#     tape.watch(x)
#     y = tf.nn.relu(x)
#     loss = tf.reduce_sum(y)
#     grads = tape.gradient(loss, x)
#     print(grads)

# import tensorflow as tf
# x = np.random.normal(size=[5, 6], scale=5.0, loc=1)
# label = np.zeros_like(x)
# label[0, 1] = 1.
# label[1, 0] = 1
# label[1, 1] = 1
# label[2, 3] = 1
# label[3, 5] = 1
# label[4, 0] = 1
# print(label)
# aa = Softmax()
# out = aa.forward(x)  # shape(5, 6)
# grad = aa.backward(label)
# print(grad)

# with tf.GradientTape() as tape:
#     x = tf.constant(x)
#     tape.watch(x)
#     y = tf.nn.softmax(x)
#     loss = tf.reduce_sum(y * label)
#     grads = tape.gradient(loss, x)
#     print(grads)

# import tensorflow as tf

# x = np.random.normal(size=[5, 6])
# aa = Log()
# out = aa.forward(x)  # shape(5, 4)
# grad = aa.backward(label)
# print(grad)

# with tf.GradientTape() as tape:
#     x = tf.constant(x)
#     tape.watch(x)
#     y = tf.math.log(x)
#     loss = tf.reduce_sum(y * label)
#     grads = tape.gradient(loss, x)
#     print(grads)

# # Final Gradient Check

# In[6]:

x = np.random.normal(size=[5, 6])   # 示例：生成 5 个样本，每个样本 6 维特征
# 初始化网络参数
label = np.zeros_like(x)            # 创建了一个与 x 形状相同的全零标签矩阵
# 手动设置每个样本的类别标签
label[0, 1] = 1.
label[1, 0] = 1
label[2, 3] = 1
label[3, 5] = 1
label[4, 0] = 1
# 重新生成输入数据（覆盖之前的x，保持代码块独立性）

x = np.random.normal(size=[5, 6])   # 5 个样本，每个样本 6 维特征
W1 = np.random.normal(size=[6, 5])  # 第一层权重 (6→5)
W2 = np.random.normal(size=[5, 6])  # 第二层权重 (5→6)

mul_h1 = Matmul()                   # 第一层矩阵乘法
mul_h2 = Matmul()                   # 第二层矩阵乘法
relu = Relu()                       # ReLU 激活函数
softmax = Softmax()                 # Softmax归一化：将输出转换为概率分布
log = Log()                         # 对数函数
# 手动实现的前向传播过程：
h1 = mul_h1.forward(x, W1)  # shape(5, 4)
h1_relu = relu.forward(h1)  # 对第一层输出h1应用ReLU激活函数（保留正值，负值置0）
h2 = mul_h2.forward(h1_relu, W2)
h2_soft = softmax.forward(h2) # 将logits转换为概率分布（[5,6]）
h2_log = log.forward(h2_soft) # 对经过 softmax 处理后的输出 h2_soft 进行对数变换
# 手动实现的反向传播过程（计算梯度）：
# 反向传播流程（从后向前）：
h2_log_grad = log.backward(-label)                # 计算损失梯度
h2_soft_grad = softmax.backward(h2_log_grad)      # Softmax梯度
h2_grad, W2_grad = mul_h2.backward(h2_soft_grad)  # 第二层权重梯度
h1_relu_grad = relu.backward(h2_grad)             # ReLU梯度
h1_grad, W1_grad = mul_h1.backward(h1_relu_grad)  # 第一层权重梯度
# 打印手动实现的反向传播梯度结果（h2_log的梯度）
print(h2_log_grad)
print('--' * 20)
# print(W2_grad)
# 使用 TensorFlow 自动微分验证梯度
with tf.GradientTape() as tape:
    # 将数据转换为 TensorFlow 常量
    x, W1, W2, label = tf.constant(x), tf.constant(W1), tf.constant(W2), tf.constant(label)
    # 将NumPy数组转换为TensorFlow常量（不可变张量）
    tape.watch(W1)        # 追踪 W1 的梯度
    tape.watch(W2)        # 追踪 W2 的梯度
    # 第一层线性变换：输入x与权重W1做矩阵乘法
    h1 = tf.matmul(x, W1)
    # 对第一层输出应用ReLU激活函数
    h1_relu = tf.nn.relu(h1)
    h2 = tf.matmul(h1_relu, W2) # [5,5] @ [5,6] → [5,6]
    prob = tf.nn.softmax(h2)
    log_prob = tf.math.log(prob) # 对数概率
    loss = tf.reduce_sum(label * log_prob)
    # 计算负对数似然损失(Negative Log Likelihood Loss)
    grads = tape.gradient(loss, [W1, W2]) # 返回[W1_grad, W2_grad]
    print("W1 Gradient Check:", grads[0].numpy())
    print("W2 Gradient Check:", grads[1].numpy())

# ## 建立模型

# In[10]:

class myModel:
    def __init__(self):# 初始化模型参数，使用随机正态分布初始化权重矩阵
        # 权重矩阵包含偏置项，通过增加输入特征维度实现
        self.W1 = np.random.normal(size=[28 * 28 + 1, 100])  # 输入层到隐藏层，增加偏置项，W1: 连接输入层(784+1)和隐藏层(100)的权重矩阵
        self.W2 = np.random.normal(size=[100, 10])           # 输入层到隐藏层，增加偏置项，W2: 连接隐藏层(100)和输出层(10)的权重矩阵
        # 初始化各层操作对象
        self.mul_h1 = Matmul()      # 第一个矩阵乘法层(输入到隐藏层)
        self.mul_h2 = Matmul()      # 第二个矩阵乘法层(隐藏层到输出层)
        self.relu = Relu()          # ReLU激活函数层
        self.softmax = Softmax()    # Softmax激活函数层，用于分类
        self.log = Log()            # 对数运算层，用于计算对数概率

    def forward(self, x):
        x = x.reshape(-1, 28 * 28)                 # 展平图像
        bias = np.ones(shape=[x.shape[0], 1])      # 添加偏置项
        x = np.concatenate([x, bias], axis=1)      # 将偏置向量添加到输入数据中
        
        # 第一层计算：输入层 -> 隐藏层
        self.h1 = self.mul_h1.forward(x, self.W1)  # shape(5, 4),#线性变换
        self.h1_relu = self.relu.forward(self.h1)  # 应用ReLU激活函数
        # 第二层计算：隐藏层 -> 输出层
        self.h2 = self.mul_h2.forward(self.h1_relu, self.W2)  # 线性变换
        self.h2_soft = self.softmax.forward(self.h2)          # 应用Softmax函数，得到概率分布
        self.h2_log = self.log.forward(self.h2_soft)          # 计算对数概率

    def backward(self, label):
        # 第二层梯度计算
        self.h2_log_grad = self.log.backward(-label)                         # 对数层梯度
        self.h2_soft_grad = self.softmax.backward(self.h2_log_grad)          # Softmax层梯度
        self.h2_grad, self.W2_grad = self.mul_h2.backward(self.h2_soft_grad) # 计算W2梯度
        # 第一层梯度计算
        self.h1_relu_grad = self.relu.backward(self.h2_grad)                 # ReLU层梯度
        self.h1_grad, self.W1_grad = self.mul_h1.backward(self.h1_relu_grad) # 计算W1梯度


model = myModel()


# ## 计算 loss

# In[11]:

def compute_loss(log_prob, labels):
    """
    计算交叉熵损失的均值。
    
    参数:
    - log_prob: numpy.ndarray
        形状为 (N, C) 的数组，表示每个样本属于每个类别的对数概率（log-probabilities）。
        通常来自模型输出并经过 log_softmax 处理。
    - labels: numpy.ndarray
        形状为 (N, C) 的 one-hot 编码标签数组，每个样本对应一个类别分布。

    返回:
    - loss: float
        所有样本的平均交叉熵损失。
        
    数学公式:
        loss = - (1/N) * ΣΣ y_ij * log(p_ij)
    """
    return -np.mean(np.sum(labels * log_prob, axis=1))


def compute_accuracy(log_prob, labels):
    """
    计算模型预测准确率。
    
    参数:
    - log_prob: 对数概率，通常是模型输出经过 log_softmax 后的结果，形状为 (batch_size, num_classes)
    - labels: 真实标签，可以是 one-hot 编码形式，形状为 (batch_size, num_classes)
    
    返回:
    - 准确率（正确预测的比例）
    """
    
    # 获取每个样本的预测类别编号（取对数概率最大的类别作为预测）
    predictions = np.argmax(log_prob, axis=1)
    
    # 获取真实标签对应的类别编号（如果是 one-hot 编码，也用 argmax 转换为类别编号）
    truth = np.argmax(labels, axis=1)
    
    # 比较预测结果与真实标签，计算正确率（布尔值数组的均值即为准确率）
    return np.mean(predictions == truth)


# 单步训练函数
def train_one_step(model, x, y):
    # 前向传播：计算模型的输出
    model.forward(x)
    # 反向传播：计算梯度
    model.backward(y)
    # 使用梯度下降法更新权重，学习率为 1e-5
    model.W1 -= 1e-5 * model.W1_grad
    model.W2 -= 1e-5 * model.W2_grad
    # 计算损失值
    loss = compute_loss(model.h2_log, y)
    # 计算准确率
    accuracy = compute_accuracy(model.h2_log, y)
    return loss, accuracy


# 测试函数
def test(model, x, y):
    model.forward(x)                             # 执行模型的正向传播，计算预测结果
    loss = compute_loss(model.h2_log, y)         # 计算损失值
    accuracy = compute_accuracy(model.h2_log, y) # 计算准确率
    return loss, accuracy                        # 返回损失值和准确率


# ## 实际训练

# In[12]:
#定义预处理函数，从原始数据中提取图像和标签，并将标签进行 one-hot 编码
def prepare_data():
    train_data, test_data = mnist_dataset()
    train_label = np.zeros(shape=[train_data[0].shape[0], 10])
    test_label = np.zeros(shape=[test_data[0].shape[0], 10])
    train_label[np.arange(train_data[0].shape[0]), np.array(train_data[1])] = 1.
    test_label[np.arange(test_data[0].shape[0]), np.array(test_data[1])] = 1.
    return train_data[0], train_label, test_data[0], test_label


def train(model, train_data, train_label, epochs=50, batch_size=128):
    losses = []
    accuracies = []
    num_samples = train_data.shape[0]

    for epoch in tqdm(range(epochs), desc="Training"):
        # 打乱数据顺序
        indices = np.random.permutation(num_samples)
        shuffled_data = train_data[indices]
        shuffled_labels = train_label[indices]
        epoch_loss = 0
        epoch_accuracy = 0

        # 小批量训练
        for i in range(0, num_samples, batch_size):
            # 获取当前批次数据
            batch_data = shuffled_data[i:i + batch_size]
            batch_labels = shuffled_labels[i:i + batch_size]
            # 执行单步训练
            loss, accuracy = train_one_step(model, batch_data, batch_labels)
            # 累计统计量
            epoch_loss += loss * batch_data.shape[0]
            epoch_accuracy += accuracy * batch_data.shape[0]

        # 计算整个 epoch 的平均损失和准确率
        epoch_loss /= num_samples
        epoch_accuracy /= num_samples
        losses.append(epoch_loss)
        accuracies.append(epoch_accuracy)
        print(f'Epoch {epoch}: Loss {epoch_loss:.4f}; Accuracy {epoch_accuracy:.4f}')
    return losses, accuracies

if __name__ == "__main__":
    # 准备数据：加载并预处理训练和测试数据
    # train_data: 训练图像数据
    # train_label: 训练标签数据
    # test_data: 测试图像数据
    # test_label: 测试标签数据
    train_data, train_label, test_data, test_label = prepare_data()
    model = myModel()
    losses, accuracies = train(model, train_data, train_label)
    test_loss, test_accuracy = test(model, test_data, test_label)
    print(f'Test Loss {test_loss:.4f}; Test Accuracy {test_accuracy:.4f}')
