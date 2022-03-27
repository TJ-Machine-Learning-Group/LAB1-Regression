from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler

def Data_preprocessing(data_url):
    raw_df = pd.read_excel(data_url)
    data = raw_df.values[:, :-1]
    #pca=PCA(n_components=data.shape[1]-2)
    #newX = pca.fit_transform(data)
    #print(pca.explained_variance_ratio_)
    newX=data
    print("newX of boston:",newX.shape)

    target = raw_df.values[:, -1]
    print("target of boston:",target.shape)

    sc = StandardScaler()
    newX = sc.fit_transform(newX)#���Ա�����һ��Ϊ��׼��̬�ֲ�,���ݶ��½�����Ӱ�켫��
    return newX,target
