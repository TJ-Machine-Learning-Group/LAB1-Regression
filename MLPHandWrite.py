import numpy as np
import scipy.optimize as opt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))
    #return np.where(z > 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))


def sigmoid_grad(z):
    g = sigmoid(z)
    return np.multiply(g, 1 - g)


def relu(inx):  # relu激活函数
    return np.where(inx > 0, inx, 0)


def relu_grad(inx):  # relu的梯度
    return np.where(inx > 0, 1, 0)

def unroll(mat):  # 展开为一维
    vec = []
    for i in mat:
        vec.append(i.ravel())
    return np.concatenate(vec)

def rand_init(network_struct):  # 随机初始化网络参数
    theta_list = []
    for i in range(len(network_struct) - 1):
        theta_list.append((2 * np.random.rand(network_struct[i + 1], network_struct[i] + 1)) - 1)
    return (theta_list)

class MLPHandWrite(object):
    def __init__(self,network_struct,reg_const=0,learning_rate=0.003):
        self.network_struct=network_struct
        self.theta=unroll(rand_init(network_struct))
        self.reg_const=reg_const
        self.learning_rate=learning_rate

    def Activation_Function(self,inx):
        return sigmoid(inx)

    def Activation_Function_grad(self,inx):
        return sigmoid_grad(inx)
    
    def fit(self,X_train,y_train):
        m,n=X_train.shape
        X = np.column_stack((np.ones(m), X_train))  # 增广，方便与bias矩阵乘法运算
        #第一种：手写Adam
        self.My_Adam(X,y_train)

        #第二种：调库BFGS
        #Result = opt.minimize(jac=self.gradient,
        #              fun=self.cost_func,
        #              x0=self.theta,
        #              args=(X, y_train),
        #              method='BFGS'
        #              )  # 最小化loss
        #self.theta = Result.x  # x为值

        #fp=open("loss.txt","w",encoding="utf8")

        #第三种：传统方法（上课讲的方法）
        #for i in range(100):  # 10000次迭代
        #    grad = self.gradient(X, y_train) 
        #    self.theta = self.theta - self.learning_rate * grad

            #fp.write(f"\n第{i}次迭代loss值：{self.cost_func(X_train, y_train)}")
        #fp.close()

    def predict(self,X_train):
        m,n=X_train.shape
        X = np.column_stack((np.ones(m), X_train))  # 增广，方便与bias矩阵乘法运算
        return self.Forward_propagation(X)[0].T

    def score(self,X_train,y):
        m,n=X_train.shape
        X = np.column_stack((np.ones(m), X_train))  # 增广，方便与bias矩阵乘法运算
        (H,tep) = self.Forward_propagation(X) # 前向传播
        Y = np.reshape(y.T, (m, 1))
        # for i in range(m):
        #    print("Real strength: ", Y[i])
        #    print("Predicted strength: ", H[i], "\n")
        score = 1 - ((Y - H)**2).sum() / ((Y - Y.mean())**2).sum()
        return score

    def Forward_propagation(self,X):
        Theta, network_struct=self.theta,self.network_struct
        m, n = X.shape
        cur, pos = 0, 0
        theta_list = []
        for i in range(len(network_struct) - 1):
            pos = (network_struct[i] + 1) * network_struct[i + 1] + pos
            theta_list.append(Theta[cur:pos].reshape((network_struct[i + 1], network_struct[i] + 1)))  # 下一层，前一层+1
            cur = pos
        A = X
        for i in range(len(theta_list) - 1):
            A = self.Activation_Function(np.dot(A, theta_list[i].T))  # 矩阵乘法后relu激活
            A = np.column_stack((np.ones(m), A))  # 增广，方便下次运算
        H = np.dot(A, theta_list[-1].T)  # 最后一层,无需relu激活及增广
        return H,theta_list

    def cost_func(self,X, y):  
        H,theta_list = self.Forward_propagation(X) # 前向传播
        m, n = X.shape
        Y = np.reshape(y.T, (m, 1))

        Diff = H - Y
        cost = Diff * Diff
        J = (1 / (m * 2)) * np.sum(cost)  # 均方差

        # 额外计算正则化项(L2)
        L2_sum = 0
        for para in theta_list:
            theta = para[:, 1:]  # 不计算第一列的偏置
            L2_sum += np.sum(theta * theta)
        J = J + ((self.reg_const / (2 * m)) * L2_sum)
        return J

    def gradient(self,X, y):#计算梯度
        Theta, network_struct=self.theta,self.network_struct
        m, n = X.shape
        cur, pos = 0, 0
        theta_list = []
        for i in range(len(network_struct) - 1):
            pos = (network_struct[i] + 1) * network_struct[i + 1] + pos
            theta_list.append(Theta[cur:pos].reshape((network_struct[i + 1], network_struct[i] + 1)))  # 下一层，前一层+1
            cur = pos

        Y = np.reshape(y, (m, 1))
        grad_list = [np.zeros(i.shape) for i in theta_list]  # 梯度列表

        for t in range(m):  # 做m次 m为训练集样本数
            x = X[t, :]
            A = x.reshape((1, n))  # 1*(input+1)的矩阵
            A_list = [A]
            Z_list = [-1]  # 没有Z0
            for i in range(len(theta_list) - 1):
                Z = np.dot(A, theta_list[i].T)  # 转置 然后矩阵乘法得到1*k
                A = self.Activation_Function(Z)
                A = np.column_stack((np.ones(1), A))  # 增广，方便下次运算
                Z = np.column_stack((np.ones(1), Z))  # 增广，方便下次运算
                A_list.append(A)
                Z_list.append(Z)

            h = np.dot(A, theta_list[-1].T)  # 最后一层 完成前向传播
            diff = h - Y[t, :]  # 与对应样本求差
            grad_list[-1] = grad_list[-1] + np.dot(diff, A_list[-1])  # 最后一层

            tepD = diff
            for i in range(len(grad_list) - 2, -1, -1):
                tepD = (np.dot(tepD, theta_list[i + 1]) * self.Activation_Function_grad(Z_list[i + 1]))[:, 1:]
                D = np.dot(tepD.T, A_list[i])
                grad_list[i] = grad_list[i] + D  # 增广的第一行应该去掉

        for i in range(len(grad_list)):
            grad_list[i] = grad_list[i] / m

        # 反向传播，额外计算正则化项(L2)
        for i in range(len(theta_list)):
            para = theta_list[i]
            para[:, 0] = np.zeros((para[:, 0]).shape)  # 不计算第一列的偏置
            grad_list[i] = grad_list[i] + (self.reg_const / m) * para
        vec = unroll(grad_list)#展平
        return vec

    def My_Adam(self, X, y, alpha=0.5, beta1=0.9, beta2=0.999,epsilon=1e-8, max_iter=100):
        m = 0
        v = 0
        for it in range(1, max_iter):
            g = self.gradient(X, y)
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * (g**2)
            m2 = m / (1 - beta1**it)
            v2 = v / (1 - beta2**it)
            update_theta=alpha * (m2 / (np.sqrt(v2) + epsilon))
            self.theta = self.theta - update_theta
            #if(update_theta.max()<epsilon or update_theta.min()>-epsilon):
            #    break;
'''
from sklearn.preprocessing import StandardScaler

data_url = "./Concrete_Data.xls"
raw_df = pd.read_excel(data_url)
data = raw_df.values[:, :-1]
print("data of boston:", data.shape)
target = raw_df.values[:, -1]
print("target of boston:", target.shape)
sc = StandardScaler()
data = sc.fit_transform(data)  #将自变量归一化为标准正态分布,对神经网络影响极大

m, fnum = data.shape  # m:样本个数
network_struct = (fnum, 9, 5, 1)  # 隐藏层为 (9, 5)
lamb = 0  # L2正则化系数
learning_rate = 0.003

#X_scaled = preprocessing.scale(X)  # 特征缩放
X_final = np.column_stack((np.ones(m), data))  # 增广，方便与bias矩阵乘法运算

X_train, X_test, y_train, y_test = ms.train_test_split(X_final, target, test_size=0.20)  # 训练集 测试集

theta_list = rand_init(network_struct)  # 初始化神经网络参数
theta = unroll(theta_list)  # 展开成一维向量

print("Training Neural Network...\n\n")


'''
