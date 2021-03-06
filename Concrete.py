from sklearn.metrics import mean_squared_error,mean_absolute_error,max_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge,ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from DecisionTreeRegressorHandWrite import DecisionTreeRegressorHandWrite
from MLPHandWrite import MLPHandWrite
from LinearModelHandWrite import LinearRegressionHandWrite,LassoHandWrite,RidgeHandWrite,ElasticNetHandWrite
from RandomForestHandWrite import myRandomForest
from Data_preprocessing import Data_preprocessing
from Regression import Regression,Draw
#import numpy as np
import copy
def main(data_url):
    data,target=Data_preprocessing(data_url)

    #实例化回归模型
    # Linear Regression
    lr_skl = LinearRegression()
    lr_handwriting = LinearRegressionHandWrite()
    # Lasso Regression
    lasso_skl = Lasso()
    lasso_handwriting= LassoHandWrite()
    # Ridge Regression
    ridge_skl = Ridge()
    ridge_handwriting=RidgeHandWrite()
    #ElasticNet Regression
    enet_skl=ElasticNet()
    enet_handwriting=ElasticNetHandWrite()

    # Decision Trees
    dtr_skl_mse = DecisionTreeRegressor()
    dtr_skl_fmse = DecisionTreeRegressor(criterion="friedman_mse")
    dtr_skl_mae = DecisionTreeRegressor(criterion='absolute_error')
    dtr_handwriting = DecisionTreeRegressorHandWrite()
    # Random Forest Regressor
    rfr_skl_se = RandomForestRegressor(n_estimators=300)
    rfr_skl_ae = RandomForestRegressor(n_estimators=300,criterion="absolute_error")
    rfr_skl_p = RandomForestRegressor(n_estimators=300,criterion="poisson")

    rfr_handwriting_se = myRandomForest(random_state=2, n_estimators=10, max_features=4, max_depth=12, min_change=0.001,min_samples_leaf=1, min_samples_split=2)
    rfr_handwriting_ae = myRandomForest(criterion="MAE", random_state=2, n_estimators=10, max_features=4, max_depth=12, min_change=0.001,min_samples_leaf=1, min_samples_split=2)
    
    # Multi-Layer Perceptron
    mlp_skl = MLPRegressor(hidden_layer_sizes=(100,70),max_iter=1800)
    mlp_handwriting=MLPHandWrite(network_struct=(data.shape[1],9,5,1),reg_const=1)
    
    models = [lr_skl, lasso_skl, ridge_skl,lr_handwriting,lasso_handwriting, ridge_handwriting,enet_skl,enet_handwriting,
             dtr_skl_mse, dtr_skl_fmse, dtr_skl_mae, dtr_handwriting, 
             rfr_skl_se,rfr_skl_ae,rfr_skl_p,rfr_handwriting_se,rfr_handwriting_ae,
             mlp_skl,mlp_handwriting]
    names = ["Linear Regression from sklearn", "Lasso Regression from sklearn", "Ridge Regression from sklearn", 
            "Linear Regression writing by hand", "Lasso Regression writing by hand", "Ridge Regression writing by hand", 
            "ElasticNet Regression from sklearn","ElasticNet Regression writing by hand",
            "Decision Tree Regressor from sklearn(squared_error)", "Decision Tree Regressor from sklearn(friedman_mse)",
            "Decision Tree Regressor from sklearn(absolute_error)","Decision Tree Regressor writing by hand", 
            "Random Forest Regressor from sklearn(squared_error)","Random Forest Regressor from sklearn(absolute_error)",
            "Random Forest Regressor from sklearn(poisson)","Random Forest Regressor written by hand(squared_error)",
            "Random Forest Regressor written by hand(absolute_error)","Multi-Layer Perceptron Regressor from sklearn",
            "Multi-Layer Perceptron writing by hand"]

    mses = []
    R2_score=[]
    times=[]
    import time
 
    for i in range(len(models)):
        #参数为5折验证，测试集占20%
        print(names[i])
        tic = time.time()
        R2_score.append(Regression(models[i],data,target,splits=5,size=0.2,model_name=names[i]))
        toc = time.time()
        mses.append(mean_squared_error(target, models[i].predict(data)))
        times.append((toc-tic)*1000)
    timeLabel = copy.deepcopy(times)
    timeLabel[-1] = ">1e6"
    times[-1] = 0#手写的多层感知机没有训练，而是直接加载了模型（因为训练速度太慢）
    # times[-1]=">1e6"

    Draw(names,mses,title="MSE")
    Draw(names,R2_score,title="R2_score")
    Draw(names,times,title="Time/ms",labels=timeLabel)

if __name__=='__main__':
    url="./Concrete_Data.xls"
    main(url)
