
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit
#�ع麯��
def Regression(model,boston_data,boston_target,splits,size,model_name="Mymodel"):
   #n�۽�����֤���������ݼ�˳��
        shuffle = ShuffleSplit(n_splits=splits, test_size=size, random_state=7)
        n_fold = 1
        score_all = 0
        X = boston_data
        Y = boston_target
        #ѵ������ѭ��
        for train_indices, test_indices in shuffle.split(boston_data):
            #��ȡ���۵�����
            x_train = X[train_indices]
            y_train = Y[train_indices]
            x_test = X[test_indices]
            y_test = Y[test_indices]
            #ģ��ѵ��
            model.fit(x_train,y_train)
            #�������ϵ��R^2
            score = model.score(x_test, y_test)
            #����
            result = model.predict(x_test)
            plt.scatter(y_test, result)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(model_name)
            plt.show()
            #��ͼ
            #plt.plot(np.arange(len(result)), y_test,label='true value')
            #plt.plot(np.arange(len(result)),result,label='predict value')
            #plt.legend(loc='upper right')
            #plt.show()
            
            print('fold {}/{},score(R^2)={}'.format(n_fold,splits,score))
            score_all += score
            n_fold += 1
        print("average score(R^2):",score_all/splits)
