import numpy as np

def rand_init(theta_num):  # 随机初始化模型参数
    return (2 * np.random.rand(theta_num)) - 1 
def cost_func(Theta,X, y,reg_const=0,reg_const2=0,L=0):#计算梯度,X已增广过,L为0则不正则化,否则为L1,L2
    m, n = X.shape
    theta=Theta
    J=0
    for t in range(m):#每个样本
        x=X[t,:]
        h=np.matmul(x,theta)
        diff=h-y[t]
        J=J+diff*diff
    J=(1.0/(m*2))*J# 均方差
    
    # 额外计算正则化项(L1/2)
    if(L==1):
        J = J + reg_const * np.sum(theta)
    if(L==2):
        J = J + reg_const * np.sum(theta*theta) / 2.0
    if(L==3):
        J = J +reg_const * np.sum(theta)+ reg_const * np.sum(theta*theta) / 2.0

    return J

def gradient(Theta,X, y,reg_const=0,reg_const2=0,L=0):#计算梯度,X已增广过,L为0则不正则化,否则为L1,L2
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
        grad = grad + reg_const * np.where(theta!=0,np.absolute(theta)/theta,0)#L1导数为绝对值除以自身
    if(L==2):
        grad = grad +reg_const * theta
    if(L==3):
        grad=grad+reg_const2 * theta+reg_const * np.where(theta!=0,np.absolute(theta)/theta,0)
    return grad

# Linear Regression
class LinearRegressionHandWrite(object):
    def __init__(self,batch_size=128,learning_rate=0.00001):
        self.coef = None
        self.learning_rate = learning_rate
        #self.batch_size=batch_size

    def fit(self, data, target,isgrad=False):
        m,n=data.shape
        self.coef=rand_init(n+1)#要加上bias
        x_train=np.column_stack((np.ones(m),data))
        if isgrad:
            #fp=open("linear_loss_lg.txt","w",encoding="utf8")
            #cur=0
            for i in range(400):  # 1000次迭代
                grad = gradient(self.coef,x_train, target) 
                #grad = gradient(self.coef,x_train[cur:cur+self.batch_size], target) 
                #cur=cur+self.batch_size
                #if cur>=m:
                #    cur=0
                #fp.write(f"\n第{i}轮,loss={np.log10(cost_func(self.coef,x_train, target))}")
                #print(np.abs(grad.min()),np.abs(grad.max()))
                #if max(np.abs(grad.min()),np.abs(grad.max()))<10:
                #    break
                self.coef = self.coef - self.learning_rate * grad
            #fp.close()
        else:
            self.coef = np.matmul(np.matmul(np.linalg.inv(np.matmul(x_train.T, x_train)), x_train.T), target)#最小二乘法

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
    def __init__(self,learning_rate=0.01,reg_const=0.1):
        self.coef = None
        self.learning_rate = learning_rate
        self.reg_const=reg_const

    def fit(self, data, target):
        m,n=data.shape
        self.coef=rand_init(n+1)#要加上bias
        #fp=open("lasso_loss.txt","w",encoding="utf8")
        x_train=np.column_stack((np.ones(m),data))
        for i in range(400):  # 1000次迭代
            grad = gradient(self.coef,x_train, target,reg_const=self.reg_const,L=1) 
            self.coef = self.coef - self.learning_rate * grad
            #fp.write(f"\n第{i}轮,loss={cost_func(self.coef,x_train, target,reg_const=self.reg_const,L=1)}")
        #fp.close()

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
    def __init__(self,learning_rate=0.01,reg_const=0.1):
        self.coef = None
        self.learning_rate = learning_rate
        self.reg_const=reg_const

    def fit(self, data, target,isgrad=False):
        m,n=data.shape
        self.coef=rand_init(n+1)#要加上bias
        x_train=np.column_stack((np.ones(m),data))
        if isgrad:
            #fp= open("ridge_loss.txt","w",encoding="utf8")
            for i in range(400):  # 1000次迭代
                grad = gradient(self.coef,x_train, target,reg_const=self.reg_const,L=2) 
                self.coef = self.coef - self.learning_rate * grad
                #fp.write(f"\n第{i}轮,loss={cost_func(self.coef,x_train, target,reg_const=self.reg_const,L=2)}")
            #fp.close()
        else:
            self.coef = np.matmul(np.matmul(np.linalg.inv(np.matmul(x_train.T, x_train)+self.reg_const*np.eye(x_train.shape[1])), x_train.T), target)

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

# ElasticNet Regression
class ElasticNetHandWrite():
    def __init__(self,learning_rate=0.01,reg_const1=0.1,reg_const2=0.1):
        self.coef = None
        self.learning_rate = learning_rate
        self.reg_const1=reg_const1
        self.reg_const2=reg_const2

    def fit(self, data, target):
        m,n=data.shape
        #fp= open("ElasticNetHandWrite_loss_0.01_0.01_0.01.txt","w",encoding="utf8")
        self.coef=rand_init(n+1)#要加上bias
        x_train=np.column_stack((np.ones(m),data))
        for i in range(400):  # 1000次迭代
            #print(self.coef)
            grad = gradient(self.coef,x_train, target,reg_const=self.reg_const1,reg_const2=self.reg_const2,L=3) 
            self.coef = self.coef - self.learning_rate * grad
            #fp.write(f"\n第{i}轮,loss={cost_func(self.coef,x_train, target,reg_const=self.reg_const1,reg_const2=self.reg_const2,L=3)}")
        #fp.close()

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
from Data_preprocessing import Data_preprocessing
from sklearn.linear_model import ElasticNet
from Regression import Regression
import pandas as pd
if __name__ == "__main__":
    data_url="./Concrete_Data.xls"
    data,target=Data_preprocessing(data_url)
    #raw_df = pd.read_excel(data_url)
    #data = raw_df.values[:, :-1]
    #target = raw_df.values[:, -1]

    #model = LinearRegressionHandWrite()
    #model= LassoHandWrite()
    #model=RidgeHandWrite()
    #model=ElasticNetHandWrite()
    model=ElasticNet()
    Regression(model, data, target, splits=1, size=0.2,model_name="ElasticNet from sklearn",isshow=True)
