from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from DecisionTreeRegressorHandWrite import DecisionTreeRegressorHandWrite
from MLPHandWrite import MLPHandWrite
from LinearModelHandWrite import LinearRegressionHandWrite,LassoHandWrite,RidgeHandWrite
from Data_preprocessing import Data_preprocessing
from Regression import Regression

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
    
    # Decision Trees
    dtr_skl = DecisionTreeRegressor()
    dtr_handwriting = DecisionTreeRegressorHandWrite()
    # Random Forest Regressor
    rfr_skl = RandomForestRegressor(n_estimators=300)
    
    # Multi-Layer Perceptron
    mlp_skl = MLPRegressor(hidden_layer_sizes=(100,70),max_iter=1800)
    mlp_handwriting=MLPHandWrite(network_struct=(data.shape[1],9,5,1),reg_const=1)
    
    models = [lr_skl, lasso_skl, ridge_skl,lr_handwriting,lasso_handwriting,ridge_handwriting,
             dtr_skl, dtr_handwriting, rfr_skl,mlp_skl,mlp_handwriting]
    names = ["Linear Regression from sklearn", "Lasso Regression from sklearn", "Ridge Regression from sklearn", 
            "Linear Regression writing by hand", "Lasso Regression writing by hand", "Ridge Regression writing by hand", 
            "Decision Tree Regressor from sklearn", "Decision Tree Regressor writing by hand", 
            "Random Forest Regressor from sklearn","Multi-Layer Perceptron Regressor from sklearn",
            "Multi-Layer Perceptron writing by hand"]
    
    for i in range(len(models)):
        #参数为5折验证，测试集占20%
        print(names[i])
        Regression(models[i],data,target,splits=1,size=0.2,model_name=names[i])

if __name__=='__main__':
    url="./Concrete_Data.xls"
    main(url)
