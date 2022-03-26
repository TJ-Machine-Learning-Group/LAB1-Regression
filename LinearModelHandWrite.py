from mimetypes import init
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit


def rand_init(theta_num):  # 随机初始化模型参数
    return (2 * np.random.rand(theta_num)) - 1 

def gradient(Theta,X, y,reg_const=0,L=0):#计算梯度,X已增广过,L为0则不正则化,否则为L1,L2
    m, n = X.shape
    theta=Theta
    #theta=np.reshape(Theta,(n,1))
    grad=np.zeros(n)
    for t in range(m):#每个样本
        x=X[t,:]
        h=np.matmul(x,theta)
        diff=h-y[t]
        grad=grad+diff*x
        
    grad = grad / m #除以m
    # 反向传播，额外计算正则化项(L1)
    if(L==1):
        grad = grad + reg_const * np.where(grad!=0,np.absolute(grad)/grad,0)#L1导数为绝对值除以自身
    if(L==2):
        grad = grad + reg_const * grad
    return grad

# Linear Regression
class LinearRegressionHandWrite(object):
    def __init__(self,learning_rate=1e-7):
        self.coef = None
        self.learning_rate = learning_rate

    def fit(self, data, target):
        m,n=data.shape
        self.coef=rand_init(n+1)#要加上bias
        x_train=np.column_stack((np.ones(m),data))
        for _ in range(10000):  # 1000次迭代
            grad = gradient(self.coef,x_train, target) 
            #print(np.abs(grad.min()),np.abs(grad.max()))
            #if max(np.abs(grad.min()),np.abs(grad.max()))<10:
            #    break
            self.coef = self.coef - self.learning_rate * grad

        #self.coef = np.matmul(np.matmul(np.linalg.inv(np.matmul(x_train.T, x_train)), x_train.T), target)#最小二乘法

    def score(self, data, target):
        m,n=data.shape
        X = np.column_stack((np.ones(m), data))  # 增广，方便与bias矩阵乘法运算
        H=np.matmul(X,np.reshape(self.coef,(n+1,1)))
        Y = np.reshape(target, (m, 1))
        # for i in range(m):
        #    print("Real strength: ", Y[i])
        #    print("Predicted strength: ", H[i], "\n")
        score = 1 - ((Y - H)**2).sum() / ((Y - Y.mean())**2).sum()
        return score

    def predict(self,data):
        m,n=data.shape
        X = np.column_stack((np.ones(m), data))  # 增广，方便与bias矩阵乘法运算
        H=np.matmul(X,np.reshape(self.coef,(n+1,1)))
        return H

# Lasso Regression
class LassoHandWrite():
    def __init__(self,learning_rate=1e-6,reg_const=1):
        self.coef = None
        self.learning_rate = learning_rate
        self.reg_const=reg_const

    def fit(self, data, target):
        m,n=data.shape
        self.coef=rand_init(n+1)#要加上bias
        x_train=np.column_stack((np.ones(m),data))
        for _ in range(3000):  # 1000次迭代
            grad = gradient(self.coef,x_train, target,reg_const=self.reg_const,L=1) 
            self.coef = self.coef - self.learning_rate * grad

        #self.coef = np.matmul(np.matmul(np.linalg.inv(np.matmul(x_train.T, x_train)), x_train.T), target)#最小二乘法

    def score(self, data, target):
        m,n=data.shape
        X = np.column_stack((np.ones(m), data))  # 增广，方便与bias矩阵乘法运算
        H=np.matmul(X,np.reshape(self.coef,(n+1,1)))
        Y = np.reshape(target, (m, 1))
        # for i in range(m):
        #    print("Real strength: ", Y[i])
        #    print("Predicted strength: ", H[i], "\n")
        score = 1 - ((Y - H)**2).sum() / ((Y - Y.mean())**2).sum()
        return score

    def predict(self,data):
        m,n=data.shape
        X = np.column_stack((np.ones(m), data))  # 增广，方便与bias矩阵乘法运算
        H=np.matmul(X,np.reshape(self.coef,(n+1,1)))
        return H

# Ridge Regression
class RidgeHandWrite():
    def __init__(self,learning_rate=1e-7,reg_const=1):
        self.coef = None
        self.learning_rate = learning_rate
        self.reg_const=reg_const

    def fit(self, X, target):
        m,n=X.shape
        self.coef=rand_init(n+1)#要加上bias
        data=np.column_stack((np.ones(m),X))
        for _ in range(10000):  # 1000次迭代
            grad = gradient(self.coef,data, target,reg_const=self.reg_const,L=2) 
            self.coef = self.coef - self.learning_rate * grad
        #self.coef = np.matmul(np.matmul(np.linalg.inv(np.matmul(data.T, data)+self.reg_const*np.eye(data.shape[1])), data.T), target)

    def score(self, data, target):
        m,n=data.shape
        X = np.column_stack((np.ones(m), data))  # 增广，方便与bias矩阵乘法运算
        H=np.matmul(X,np.reshape(self.coef,(n+1,1)))
        Y = np.reshape(target, (m, 1))
        # for i in range(m):
        #    print("Real strength: ", Y[i])
        #    print("Predicted strength: ", H[i], "\n")
        score = 1 - ((Y - H)**2).sum() / ((Y - Y.mean())**2).sum()
        return score

    def predict(self,data):
        m,n=data.shape
        X = np.column_stack((np.ones(m), data))  # 增广，方便与bias矩阵乘法运算
        H=np.matmul(X,np.reshape(self.coef,(n+1,1)))
        return H


def regression(model, data, target, splits, size):
    # n折交叉验证并打乱数据集顺序
    shuffle = ShuffleSplit(n_splits=splits, test_size=size, random_state=7)
    n_fold = 1
    score_all = 0
    x = data
    y = target
    # 训练测试循环
    for train_indices, test_indices in shuffle.split(data):
        # 获取此折的数据
        x_train = x[train_indices]
        y_train = y[train_indices]
        x_test = x[test_indices]
        y_test = y[test_indices]
        # 模型训练
        model.fit(x_train, y_train)
        # 计算决定系数R^2
        score = model.score(x_test, y_test)
        # 测试
        result = model.predict(x_test)
        print(model.coef,score,sep = '\n')
        print('fold {}/{},score(R^2)={}'.format(n_fold, splits, score))
        score_all += score
        n_fold += 1
    print("average score(R^2):", score_all / splits)


if __name__ == "__main__":
    data_url = "./Concrete_Data.xls"
    raw_df = pd.read_excel(data_url)
    data = raw_df.values[:, :-1]
    target = raw_df.values[:, -1]

    #model = LinearRegressionHandWrite()
    model= LassoHandWrite()

    #model=RidgeHandWrite()
    regression(model, data, target, splits=5, size=0.2)
