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
    newX = sc.fit_transform(newX)#将自变量归一化为标准正态分布,对梯度下降方法影响极大
    return newX,target

def outlier_test(data, column, method=None, z=2):
    
    if method == None:
        print(f'以 {column} 列为依据，使用 上下截断点法(iqr) 检测异常值...')
        print('=' * 70)
        column_iqr = np.quantile(data[column], 0.75) - np.quantile(data[column], 0.25)
        (q1, q3) = np.quantile(data[column], 0.25), np.quantile(data[column], 0.75)
        upper, lower = (q3 + 1.5 * column_iqr), (q1 - 1.5 * column_iqr)
        outlier = data[(data[column] <= lower) | (data[column] >= upper)]
        print(f'第一分位数: {q1}, 第三分位数：{q3}, 四分位极差：{column_iqr}')
        print(f"上截断点：{upper}, 下截断点：{lower}")
        return outlier, upper, lower
    
    if method == 'z':
        
        print(f'以 {column} 列为依据，使用 Z 分数法，z 分位数取 {z} 来检测异常值...')
        print('=' * 70)    
        mean, std = np.mean(data[column]), np.std(data[column])
        upper, lower = (mean + z * std), (mean - z * std)
        print(f"取 {z} 个 Z分数：大于 {upper} 或小于 {lower} 的即可被视为异常值。")
        print('=' * 70)
        outlier = data[(data[column] <= lower) | (data[column] >= upper)]
        return outlier, upper, lower
