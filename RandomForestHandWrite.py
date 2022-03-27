import numpy as np
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split as tt_split
from sklearn.metrics import r2_score

import warnings 
warnings.filterwarnings('ignore')


# 计算方差
class squaredError:
    def __init__(self) -> None:
        pass
    
    def value(self,dataSet):
        #print((dataSet[:,-1])*shape(dataSet)[0])
        return np.var(dataSet[:,-1])*np.shape(dataSet)[0]

# 计算绝对误差
class absoluteError:
    def __init__(self) -> None:
        pass
    
    def value(self,dataSet):
        M = np.mean(dataSet[:,-1])
        mean= np.mat(np.ones((len(dataSet[:,-1]),1))*M)
        return np.sum(list(map(abs,list(dataSet[:,-1]-M))))

class myRandomForest(object):
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
        for i in range(np.shape(dataSet)[0]):
            if dataSet[i][feature] == value and dataSet[i+1][feature] != value:
                return dataSet[i+1:, :], dataSet[0:i+1, :]
    
    # 选取特征边界
    def selectBestFeature(self, dataSet):
        #计算特征的数目
        feature_num=dataSet.shape[1]-1
        features=np.random.choice(feature_num,self.max_features,replace=False)
        # 最好分数
        bestScore=np.inf;
        # 最优特征
        bestfeature=0;
        # 最优特征的分割值
        bestValue=0;
        curScore=self.criterion.value(dataSet)
        # 判断样本数量是否足够
        if np.shape(dataSet)[0] < self.min_samples_split or np.shape(dataSet)[0] < self.min_samples_leaf:
            return None,self.getMean(dataSet)
        for feature in features:
            dataSet = dataSet[dataSet[:,feature].argsort()]
            # 控制叶子节点数目
            for index in range(np.shape(dataSet)[0]-1):
                # 排除重复值
                if index != np.shape(dataSet)[0]-1 and dataSet[index][feature] == dataSet[index+1][feature]:
                    continue
                data0 = dataSet[0:index+1, :]
                data1 = dataSet[index+1:, :]
                if np.shape(data0)[0] < self.min_samples_leaf or np.shape(data1)[0] < self.min_samples_leaf:
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
    def __init__(self, random_state=2, n_estimators=10, max_features=4, max_depth=12, min_change = 0.001,criterion="MSE",
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
        predict=np.mat(np.zeros((l,1)))
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
        predict=np.mat(np.zeros((l,1)))
        Parallel(n_jobs=self.n_jobs, backend="threading")(delayed(self.updatePredict)(predict, tree, X) for tree in self.trees)
    #     对多棵树预测的结果取平均
        predict/=self.n_estimators
        return predict.A
    
    # 获取模型分数
    def score(self, X, target):
        return r2_score(target, self.predict(X))

from Data_preprocessing import Data_preprocessing
from Regression import Regression

if __name__=='__main__':
    data,target=Data_preprocessing("./Concrete_Data.xls")
    model = myRandomForest()
    Regression(model, data, target, splits=1, size=0.2)
