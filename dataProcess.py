import pandas as pd
import numpy as np
#from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
import matplotlib.pyplot as plt
import itertools
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from sklearn.metrics import r2_score as rs
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings

warnings.filterwarnings("ignore")#忽略输出警告
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体（Windows系统内置）
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方框的问题


df= pd.read_excel('E:\\xujing\\resoureces\\flow.xlsx',sheet_name='Sheet0',skipfooter=5,engine='openpyxl')
#筛选合法的时间
df = df[df['时间'].apply(lambda x: isinstance(x, (str, pd.Timestamp)) and pd.to_datetime(x, errors='coerce') is not None)]
df['时间']=pd.to_datetime(df['时间'],errors='coerce')

#删除转换失败的行（Nat表示无效日期）
df = df.dropna(subset=['时间'])
#设置时间为横坐标变量
df.set_index('时间',inplace=True)
#筛选所需要的数据
base_data=df.loc[df.index.year == 2020, "三峡入库流量_平均值_查询值"]
forecast_data=df.loc[(df.index.year == 2021),"三峡入库流量_平均值_查询值"]
data_stl = sm.tsa.STL(base_data).fit()  # statsmodels.tsa.api:时间序列模型和方法
data_stl.plot()
# 趋势效益
trend = data_stl.trend
# 季节效应
seasonal = data_stl.seasonal
# 随机效应
residual = data_stl.resid

#平稳性检验
def test_stationarity(timeseries, alpha):  # alpha为检验选取的显著性水平
    adf = ADF(timeseries)
    p = adf[1]  # p值
    critical_value = adf[4]["5%"]  # 在95%置信区间下的临界的ADF检验值
    test_statistic = adf[0]  # ADF统计量
    if p < alpha and test_statistic < critical_value:
        print("ADF平稳性检验结果：在显著性水平%s下，数据经检验平稳" % alpha)
        return True
    else:
        print("ADF平稳性检验结果：在显著性水平%s下，数据经检验不平稳" % alpha)
        return False
#test_stationarity(base_data,1e-3)
#转化为平稳数据
base_data_diff1=base_data.diff(1)
base_data_seasonal=base_data_diff1.diff(12)
test_stationarity(base_data_seasonal.dropna(),1e-3)

#白噪声检验

#模型定阶
def SARIMA_search(data):
    p = q = range(0, 3)
    s = [12]  # 周期为12
    d = [1]  # 做了一次季节性差分
    PDQs = list(itertools.product(p, d, q, s))  # itertools.product()得到的是可迭代对象的笛卡儿积
    pdq = list(itertools.product(p, d, q))  # list是python中是序列数据结构，序列中的每个元素都分配一个数字定位位置
    params = []
    seasonal_params = []
    results = []
    grid = pd.DataFrame()
    for param in pdq:
        for seasonal_param in PDQs:
            # 建立模型
            mod = sm.tsa.SARIMAX(data, order=param, seasonal_order=seasonal_param, \
                                 enforce_stationarity=False, enforce_invertibility=False)
            # 实现数据在模型中训练
            result = mod.fit()
            print("ARIMA{}x{}-AIC:{}".format(param, seasonal_param, result.aic))
            # format表示python格式化输出，使用{}代替%
            params.append(param)
            seasonal_params.append(seasonal_param)
            results.append(result.aic)
    grid["pdq"] = params
    grid["PDQs"] = seasonal_params
    grid["aic"] = results
    print(grid[grid["aic"] == grid["aic"].min()])
#SARIMA_search(base_data_seasonal.dropna())
model=sm.tsa.SARIMAX(base_data,order=(2,1,1),seasonal_order=(2,1,1,12))
SARIMA_m=model.fit()
print(SARIMA_m.summary())
#模型检验
fig=SARIMA_m.plot_diagnostics(figsize=(15,12))#plot_diagnostics对象允许我们快速生成模型诊断并调查任何异常行为


# 获取预测结果，自定义预测误差
def PredictionAnalysis(data, model, start, dynamic=False):
    pred = model.get_prediction(start=start, dynamic=dynamic, full_results=True)
    pci = pred.conf_int()  # 置信区间
    pm = pred.predicted_mean  # 预测值
    truth = data[start:]  # 真实值
    pc = pd.concat([truth, pm, pci], axis=1)  # 按列拼接
    pc.columns = ['true', 'pred', 'up', 'low']  # 定义列索引
    print("1、MSE:{}".format(mse(truth, pm)))
    print("2、RMSE:{}".format(np.sqrt(mse(truth, pm))))
    print("3、MAE:{}".format(mae(truth, pm)))
    return pc


# 绘制预测结果
def PredictonPlot(pc):
    plt.figure(figsize=(10, 8))
    plt.fill_between(pc.index, pc['up'], pc['low'], color='grey', \
                     alpha=0.15, label='confidence interval')  # 画出置信区间
    plt.plot(pc['true'], label='base data')
    plt.plot(pc['pred'], label='prediction curve')
    plt.legend()
    plt.show()
    return True
pred=PredictionAnalysis(base_data,SARIMA_m,'2020-03-01',dynamic=True)
PredictonPlot(pred)

forecast=SARIMA_m.get_forecast(steps=365)
fig,ax=(plt.subplots(figsize=(10,8)))
base_data.plot(ax=ax, label="base data")
forecast_data.plot(ax=ax, label="exact data")
forecast.predicted_mean.plot(ax=ax, label="forecast data")
# ax.fill_between(forecast.conf_int().index(),forecast.conf_int().iloc[:,0],\
#               forecast.conf_int().iloc[:,1],color='grey',alpha=0.15,label='confidence interval')
ax.legend(loc="best", fontsize=20)
ax.set_xlabel("时间", fontsize=20)
ax.set_ylabel("三峡入库流量_平均值_查询值", fontsize=18)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()
