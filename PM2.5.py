from datetime import datetime
from pandas import read_csv
import pandas as pd
import matplotlib.pyplot as plt

from  sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
#1.数据加载

# def parse(x):
#     return datetime.strptime(x,'%Y %m %d %H')
# dataset=read_csv('F:/worklog/BI/Lession1/L9/pm2.5/raw.csv',parse_dates=[['year','month','day','hour']],index_col=0,date_parser=parse)
#
# dataset.drop('No',axis=1,inplace=True)
# #2.列明替换
# dataset.columns=['pollution','dew','temp','press','wnd_dir','wnd_spd','snow','rain']
# dataset.index.name='date'
# #3.缺失值填充
# dataset['pollution'].fillna(0,inplace=True)
# #4.去掉第一天的数据（前24小时）,因为第一天的数据是空的
# dataset=dataset[24:]
# dataset.to_csv('pollution.csv')


#数据加载
# dataset=pd.read_csv('pollution.csv')
#对数据进行可视化

dataset=pd.read_csv('pollution.csv',index_col=0)
#一共有八个特征

#对数据进行可视化
values=dataset.values
# print(values)
# i=1
#
# for group in range(8):
#     plt.subplot(8,1,i)
#     plt.plot(values[:,group])
#     plt.title(dataset.columns[group])
#     i+=1
# plt.show()

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    将时间序列数据转换为y适用监督学习的数据
    :param data:观察序列
    :param n_in:观察数据input（x）的步长
    :param n_out:输出的步长，默认为1
    :param dropnan:
    :return:适用于监督学习的步长
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # 预测序列 (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # 拼接到一起
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # 去掉NaN行
    if dropnan:
        agg.dropna(inplace=True)
    return agg

#对特殊数据进行标签编辑
# dataset['win_dir'].value_count()
#对不同的
#需要将分类特征进行标签编码

encoder=LabelEncoder()
values[:,4]=encoder.fit_transform(values[:,4])
#设置数据类型为float32

#数据归一化
scaler=MinMaxScaler()
scaled=scaler.fit_transform(values)
print(scaled.shape)
#将数据进行贴标签，并且保存
reframed=series_to_supervised(scaled,1,1)
reframed.to_csv('reframed-1.csv')

#对数据集进行切分，80%作为训练集，20%作为测试集
values=reframed.values
#LSTM不能采用train_test_split(),因为时间序列不连续
#XGBoost可以，因为样本是相互独立的
n_train_hours=int(len(values)*0.8)
train=values[:n_train_hours,:]
test=values[n_train_hours:,:]

train_X,train_y=train[:,:-1],train[:,-1]
test_X,test_y=test[:,:-1],test[:,-1]

#转化为3D格式【样本数，时间步，特征数】
train_X.reshape((train_X.shape[0],1,train_X.shape[1]))
test_X.reshape((test_X.shape[0],1,test_X.shape[1]))

from tensorflow.keras.models import Sequential
from tensorflow.keras import Dense,LSTM

#设置网络模型

model=Sequential()
model.add(LSTM(50,input_shape=(train_X.shape[1],train_X.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse')

#模型训练

result=model.fit(train_X,train_y,epochs=10,batch_size=64,validation_data=(test_X,test_y),verbose=2,shuffle=False)

train_predict=model.predict(train_X)
test_predict=model.predict(test_X)

line1=result.history['loss']
line2=result.history['val_loss']
plt.plot(line1,label='train',c='g')
plt.plot(line2,label='test',c='r')
plt.legend(loc='best')
plt.show()


def plot_img(source_data_set,train_predict,test_predict):
    plt.plot(source_data_set[:,-1],label='real',c='b')
    plt.plot([x for x in train_predict],label='train_predict',c='g')
    temp=[None for _ in train_predict]+[x for x in test_predict]
    plt.plot(temp,label='test_predict',c='r')
    plt.legend(loc='best')
    plt.show()


plot_img(values,train_predict,test_predict)

