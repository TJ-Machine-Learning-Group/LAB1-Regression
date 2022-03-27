import numpy as np
import matplotlib.pyplot as plt
from numpy import *
from sklearn.model_selection import train_test_split as tt_split
from sklearn.metrics import r2_score
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from DecisionTreeRegressorHandWrite import *
from sklearn.metrics import mean_squared_error

import warnings 
warnings.filterwarnings('ignore')


# 计算方差
class squaredError:
    def __init__(self) -> None:
        pass
    
    def value(self,dataSet):
        #print((dataSet[:,-1])*shape(dataSet)[0])
        return np.var(dataSet[:,-1])*shape(dataSet)[0]

# 计算绝对误差
class absoluteError:
    def __init__(self) -> None:
        pass
    
    def value(self,dataSet):
        M = np.mean(dataSet[:,-1])
        mean= np.mat(ones((len(dataSet[:,-1]),1))*M)
        return np.sum(list(map(abs,list(dataSet[:,-1]-M))))

from joblib import Parallel, delayed
class myRandomForest:
    # 存放树的列表
    trees = []
    # 随机种子
    random_state = 0
    # 树的个数
    n_estimators = 10
    # 最大特征数
    max_features = 10
    # 最大深度
    max_depth = 10
    # 切分新节点所需的最小阈值
    min_change = 0.001
    # 当前树的数量
    cur_tree = 0
    # 最小分割
    min_samples_split = 0
    # 叶子内节点的最小数目
    min_samples_leaf = 0
    # 每次建树时所用的样本占总样本的比例
    sample_radio = 0.9
    # 每次建树时所并行化处理器的个数
    n_jobs = 10
    #分支策略
    criterion="MSE"
    
    # 计算y的均值
    def getMean(self,dataSet):
        return np.mean(dataSet[:,-1])
    
    # 根据特征边界划分样本
    def splitDataSet(self, dataSet,feature,value):
        dataSet = dataSet[dataSet[:,feature].argsort()]
        for i in range(shape(dataSet)[0]):
            if dataSet[i][feature] == value and dataSet[i+1][feature] != value:
                return dataSet[i+1:, :], dataSet[0:i+1, :]
    
    # 选取特征边界
    def selectBestFeature(self, dataSet):
        #计算特征的数目
        feature_num=dataSet.shape[1]-1
        features=np.random.choice(feature_num,self.max_features,replace=False)
        # 最好分数
        bestScore=inf;
        # 最优特征
        bestfeature=0;
        # 最优特征的分割值
        bestValue=0;
        curScore=self.criterion.value(dataSet)
        # 判断样本数量是否足够
        if shape(dataSet)[0] < self.min_samples_split or shape(dataSet)[0] < self.min_samples_leaf:
            return None,self.getMean(dataSet)
        for feature in features:
            dataSet = dataSet[dataSet[:,feature].argsort()]
            # 控制叶子节点数目
            for index in range(shape(dataSet)[0]-1):
                # 排除重复值
                if index != shape(dataSet)[0]-1 and dataSet[index][feature] == dataSet[index+1][feature]:
                    continue
                data0 = dataSet[0:index+1, :]
                data1 = dataSet[index+1:, :]
                if shape(data0)[0] < self.min_samples_leaf or shape(data1)[0] < self.min_samples_leaf:
                    continue;
                newS=self.criterion.value(data0)+self.criterion.value(data1)
                if bestScore>newS:
                    bestfeature=feature
                    bestValue=dataSet[index][feature]
    #                     print(bestfeature, bestValue)
                    bestScore=newS
        if (curScore-bestScore)<self.min_change: #如果误差不大就退出，说明无法分割
            return None,self.getMean(dataSet)
    #         print(bestfeature, bestValue)
        return bestfeature,bestValue

    # 搭建决策树
    def createTree(self, dataSet, max_level, flag = 0):
        if flag == 0:
            seqtree = self.cur_tree+1
            self.cur_tree = seqtree;
            #print('正在搭建第'+str(seqtree)+'棵树...\n')
        bestfeature,bestValue=self.selectBestFeature(dataSet)
        if bestfeature==None:
            if flag == 0:
                pass
                #print('第'+str(seqtree)+'棵树搭建完成！')
            return bestValue
        retTree={}
        max_level-=1
        if max_level<0:   #控制深度
            return self.getMean(dataSet)
        retTree['bestFeature']=bestfeature
        retTree['bestVal']=bestValue
        # 分割成左右两棵树
        lSet,rSet=self.splitDataSet(dataSet,bestfeature,bestValue)
        retTree['right']=self.createTree(rSet,self.max_depth,1)
        retTree['left']=self.createTree(lSet,self.max_depth,1)
        if flag == 0:
            pass
            # print('第'+str(seqtree)+'棵树搭建完成！')
        return retTree
    
    # 初始化随机森林
    def __init__(self, random_state, n_estimators, max_features, max_depth, min_change = 0.001,criterion="MSE",
                 min_samples_split = 0, min_samples_leaf = 0, sample_radio = 0.9, n_jobs = 10):
        self.trees = []
        self.random_state = random_state
        np.random.seed(self.random_state)
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_change = min_change
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.sample_radio = sample_radio
        self.n_jobs = n_jobs
        if criterion == "MAE":
            self.criterion = absoluteError()
        else:
            self.criterion = squaredError()
        
        
    # 向森林添加单棵决策树
    def addTree(self, dataSet):
        X_train, X_test, y_train, y_test = tt_split(dataSet[:,:-1], dataSet[:,-1], train_size = self.sample_radio, random_state = self.random_state)
        X_train=np.concatenate((X_train,y_train.reshape((-1,1))),axis=1)
        self.trees.append(self.createTree(X_train,self.max_depth))
    
    # 并行化搭建随机森林
    def fit(self, X, Y):   #树的个数，预测时使用的特征的数目，树的深度
        dataSet = np.concatenate((X, Y.reshape(-1,1)), axis = -1)
        self.trees=[] #确保为空再搭建森林
        self.cur_tree=0
        Parallel(n_jobs=self.n_jobs, backend="threading")(delayed(self.addTree)(dataSet) for _ in range(self.n_estimators))             
            
    #预测单个数据样本
    def treeForecast(self,tree,data):
        if not isinstance(tree,dict):
            return float(tree)
        if data[tree['bestFeature']]>tree['bestVal']:
            if type(tree['left'])=='float':
                return tree['left']
            else:
                return self.treeForecast(tree['left'],data)
        else:
            if type(tree['right'])=='float':
                return tree['right']
            else:
                return self.treeForecast(tree['right'],data) 
            
    # 单决策树预测结果
    def createForeCast(self,tree,dataSet):
        seqtree = self.cur_tree+1
        self.cur_tree = seqtree;
        # print('第'+str(seqtree)+'棵树正在预测...\n')
        l=len(dataSet)
        predict=np.mat(zeros((l,1)))
        for i in range(l):
            predict[i,0]=self.treeForecast(tree,dataSet[i,:])
         # print('第'+str(seqtree)+'棵树预测完成!')
        return predict
    
    # 更新预测值函数
    def updatePredict(self, predict, tree, X):
        predict+=self.createForeCast(tree,X)
    
    # 随机森林预测结果
    def predict(self,X):
        self.cur_tree = 0;
        l=len(X)
        predict=np.mat(zeros((l,1)))
        Parallel(n_jobs=self.n_jobs, backend="threading")(delayed(self.updatePredict)(predict, tree, X) for tree in self.trees)
    #     对多棵树预测的结果取平均
        predict/=self.n_estimators
        return predict
    
    # 获取模型分数
    def score(self, X, target):
        return r2_score(target, self.predict(X))

#回归函数
def Regression(model,boston_data,boston_target,splits,size):
   #n折交叉验证并打乱数据集顺序
        shuffle = ShuffleSplit(n_splits=splits, test_size=size, random_state=7)
        n_fold = 1
        score_all = 0
        X = boston_data
        Y = boston_target
        scores=[]
        rmses = []
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
            scores.append(score)
            
            y_pred=model.predict(x_test)
            rmses.append(np.sqrt(mean_squared_error(y_test, y_pred)))

            print('fold {}/{},score(R^2)={}'.format(n_fold,splits,score))
            score_all += score
            n_fold += 1
        # plt.plot(scores)
        # for x,y in enumerate(scores):
        #     plt.text(x, y, y, ha='center', va='bottom', fontsize=8)
        # plt.show()
        # y_pred=model.predict(X)
        # y_pred=y_pred.reshape(1,y_pred.shape[0])
        # print(y_pred.shape)
        # Y=Y.reshape(1,Y.shape[0])
        # print(Y.shape)

        # plt.scatter(Y, y_pred.A)
        # plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=2)
        # plt.ylabel("Predicted")
        # plt.xlabel("True")
        # plt.title("Random Forest Regressor")
        # plt.show()
        # x = np.arange(5) 
        # width = 0.3

        # fig, ax = plt.subplots(figsize=(10,7))
        # rects = ax.bar(x, rmses, width)
        # ax.set_ylabel('RMSE')
        # ax.set_xlabel('Models')
        # ax.set_title('loss(MSE)')
        # ax.set_xticks(x)
        # #ax.set_xticklabels(names, rotation=45)
        # for rect in rects:
        #     height = rect.get_height()
        #     ax.annotate('{:.2f}'.format(height), xy=(rect.get_x() + rect.get_width() / 2, height),
        #                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
        # fig.tight_layout()
        # plt.show()
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
    rfr_skl = RandomForestRegressor(n_estimators=100)
    
    # Multi-Layer Perceptron
    #mlp_skl = MLPRegressor(hidden_layer_sizes=(5,5),max_iter=10000)
    mlp_skl = MLPRegressor(hidden_layer_sizes=(100,70),max_iter=1800)

    rfr_hw1 = myRandomForest(random_state=2, n_estimators=10, max_features=4, max_depth=12, min_change=0.001,min_samples_leaf=1, min_samples_split=2)

    rfr_hw2 = myRandomForest(criterion="MAE", random_state=2, n_estimators=10, max_features=4, max_depth=12, min_change=0.001,min_samples_leaf=1, min_samples_split=2)
    
    models = [lr_skl, lasso_skl, ridge_skl, dtr_skl, dtr_handwriting, rfr_skl,mlp_skl,rfr_hw1,rfr_hw2]
    names = ["Linear Regression from sklearn", "Lasso Regression from sklearn", "Ridge Regression from sklearn", 
         "Decision Tree Regressor from sklearn", "Decision Tree Regressor writing by hand", "Random Forest Regressor from sklearn","Multi-Layer Perceptron Regressor from sklearn","Random Forest Regressor writing by hand(MSE)","Random Forest Regressor writing by hand(MAE)"]

    for i in range(len(models)-2,len(models)):
        #参数为5折验证，测试集占20%
        print(names[i])
        Regression(models[i],data,target,splits=5,size=0.2)

if __name__=='__main__':
    url="code/dataset/concrete/Concrete_Data.xls"
    main(url)