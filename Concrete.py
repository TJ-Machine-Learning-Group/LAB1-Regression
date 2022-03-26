# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from DecisionTreeRegressorHandWrite import DecisionTreeRegressorHandWrite
from MLPHandWrite import MLPHandWrite
#回归函数
def Regression(model,boston_data,boston_target,splits,size):
   #n折交叉验证并打乱数据集顺序
        shuffle = ShuffleSplit(n_splits=splits, test_size=size, random_state=7)
        n_fold = 1
        score_all = 0
        X = boston_data
        Y = boston_target
        #训练测试循环
        for train_indices, test_indices in shuffle.split(boston_data):
            #获取此折的数据
            x_train = X[train_indices]
            y_train = Y[train_indices]
            x_test = X[test_indices]
            y_test = Y[test_indices]
            #模型训练
            model.fit(x_train,y_train)
            #计算决定系数R^2
            score = model.score(x_test, y_test)
            #测试
            result = model.predict(x_test)

            #画图
            #plt.plot(np.arange(len(result)), y_test,label='true value')
            #plt.plot(np.arange(len(result)),result,label='predict value')
            #plt.legend(loc='upper right')
            #plt.show()
            
            print('fold {}/{},score(R^2)={}'.format(n_fold,splits,score))
            score_all += score
            n_fold += 1
        print("average score(R^2):",score_all/splits)

def Data_preprocessing(data_url):
    raw_df = pd.read_excel(data_url)
    data = raw_df.values[:, :-1]
    print("data of boston:",data.shape)
    target = raw_df.values[:, -1]
    print("target of boston:",target.shape)

    sc = StandardScaler()
    data = sc.fit_transform(data)#将自变量归一化为标准正态分布,对神经网络影响极大
    return data,target

def main(data_url):
    data,target=Data_preprocessing(data_url)

    #实例化sklearn回归模型
    # Linear Regression
    lr_skl = LinearRegression()
    # Lasso Regression
    lasso_skl = Lasso()
    # Ridge Regression
    ridge_skl = Ridge()

    # Decision Trees
    dtr_skl = DecisionTreeRegressor()
    dtr_handwriting = DecisionTreeRegressorHandWrite()
    # Random Forest Regressor
    rfr_skl = RandomForestRegressor(n_estimators=300)
    
    # Multi-Layer Perceptron
    #mlp_skl = MLPRegressor(hidden_layer_sizes=(5,5),max_iter=10000)
    mlp_skl = MLPRegressor(hidden_layer_sizes=(100,70),max_iter=100)
    mlp_handwriting=MLPHandWrite(network_struct=(data.shape[1],9,5,1),reg_const=1)
    Regression(mlp_handwriting,data,target,splits=5,size=0.2)
    
    #models = [lr_skl, lasso_skl, ridge_skl, dtr_skl, dtr_handwriting, rfr_skl,mlp_skl,mlp_handwriting]
    #names = ["Linear Regression from sklearn", "Lasso Regression from sklearn", "Ridge Regression from sklearn", 
    #     "Decision Tree Regressor from sklearn", "Decision Tree Regressor writing by hand", 
    #     "Random Forest Regressor from sklearn","Multi-Layer Perceptron Regressor from sklearn",
    #     "Multi-Layer Perceptron writing by hand"]
    #
    #for i in range(len(models)):
    #    #参数为5折验证，测试集占20%
    #    print(names[i])
    #    Regression(models[i],data,target,splits=5,size=0.2)

if __name__=='__main__':
    url="./Concrete_Data.xls"
    main(url)
