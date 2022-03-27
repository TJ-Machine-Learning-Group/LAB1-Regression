import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit
#回归函数
def Regression(model,boston_data,boston_target,splits,size,model_name="Mymodel"):
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
            plt.scatter(y_test, result)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(model_name)
            plt.show()
            #画图
            #plt.plot(np.arange(len(result)), y_test,label='true value')
            #plt.plot(np.arange(len(result)),result,label='predict value')
            #plt.legend(loc='upper right')
            #plt.show()
            
            print('fold {}/{},score(R^2)={}'.format(n_fold,splits,score))
            score_all += score
            n_fold += 1
        print("average score(R^2):",score_all/splits)

def autolabel(rects,ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.2f}'.format(height), xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

def Draw(names,mses):
    x = np.arange(len(names)) 
    fig, ax = plt.subplots(figsize=(12,8))
    ax.set_ylabel('MSE')
    ax.set_xlabel('Models')
    ax.set_title('MSE with Different Algorithms')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=270)
    width = 0.1
    rects = ax.bar(x, mses, width)
    autolabel(rects,ax)
    fig.tight_layout()
    plt.show()
