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

def outlier_test(data, column, method=None, z=2):
    
    if method == None:
        print(f'�� {column} ��Ϊ���ݣ�ʹ�� ���½ضϵ㷨(iqr) ����쳣ֵ...')
        print('=' * 70)
        column_iqr = np.quantile(data[column], 0.75) - np.quantile(data[column], 0.25)
        (q1, q3) = np.quantile(data[column], 0.25), np.quantile(data[column], 0.75)
        upper, lower = (q3 + 1.5 * column_iqr), (q1 - 1.5 * column_iqr)
        outlier = data[(data[column] <= lower) | (data[column] >= upper)]
        print(f'��һ��λ��: {q1}, ������λ����{q3}, �ķ�λ���{column_iqr}')
        print(f"�Ͻضϵ㣺{upper}, �½ضϵ㣺{lower}")
        return outlier, upper, lower
    
    if method == 'z':
        
        print(f'�� {column} ��Ϊ���ݣ�ʹ�� Z ��������z ��λ��ȡ {z} ������쳣ֵ...')
        print('=' * 70)    
        mean, std = np.mean(data[column]), np.std(data[column])
        upper, lower = (mean + z * std), (mean - z * std)
        print(f"ȡ {z} �� Z���������� {upper} ��С�� {lower} �ļ��ɱ���Ϊ�쳣ֵ��")
        print('=' * 70)
        outlier = data[(data[column] <= lower) | (data[column] >= upper)]
        return outlier, upper, lower
