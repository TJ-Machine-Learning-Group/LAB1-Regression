import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

#回归函数
def regression(model,boston_data,boston_target,splits,size):
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
            plt.plot(np.arange(len(result)), y_test,label='true value')
            plt.plot(np.arange(len(result)),result,label='predict value')
            plt.legend(loc='upper right')
            #plt.show()

            print('fold {}/{},score(R^2)={}'.format(n_fold,splits,score))
            score_all += score
            n_fold += 1
        print("average score(R^2):",score_all/splits)

def main():
    data_url = "./dataset/concrete/Concrete_Data.xls"
    raw_df = pd.read_excel(data_url)
    data = raw_df.values[:, :-1]
    print("data of boston:",data.shape)
    target = raw_df.values[:, -1]
    print("target of boston:",target.shape)

    sc = StandardScaler()
    data = sc.fit_transform(data)#将自变量归一化为标准正态分布


    #实例化线性回归模型
    # Linear Regression
    lr = LinearRegression()
    # Lasso Regression
    lasso = Lasso()
    # Ridge Regression
    ridge = Ridge()

    # Decision Trees
    dtr = DecisionTreeRegressor()
    # Random Forest Regressor
    rfr = RandomForestRegressor(n_estimators=100)
    
    # Multi-Layer Perceptron
    mlp = MLPRegressor(hidden_layer_sizes=(100,70),max_iter=1800)

    models = [lr, lasso, ridge, dtr, rfr,mlp]
    names = ["Linear Regression", "Lasso Regression", "Ridge Regression", 
         "Decision Tree Regressor", "Random Forest Regressor","Multi-Layer Perceptron Regressor"]
    for i in range(len(models)):
        #参数为5折验证，测试集占20%
        print(names[i])
        regression(mlp,data,target,splits=5,size=0.2)

if __name__=='__main__':
    main()
