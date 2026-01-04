import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from linear_regression import LinearRegression

data = pd.read_csv('../data/world-happiness-report-2017.csv')     #修改

# 得到训练和测试数据
train_data = data.sample(frac = 0.8)    #修改
test_data = data.drop(train_data.index)
#总的数据集-训练数据=测试数据

input_param_name = 'Economy..GDP.per.Capita.'
output_param_name = 'Happiness.Score'

x_train = train_data[[input_param_name]].values
y_train = train_data[[output_param_name]].values

x_test = test_data[input_param_name].values
y_test = test_data[output_param_name].values

#数据散点图构建
plt.scatter(x_train,y_train,label='Train data')  #训练点
plt.scatter(x_test,y_test,label='test data')   #测试点
plt.xlabel(input_param_name)   #X轴名称
plt.ylabel(output_param_name)  #Y轴名称
plt.title('Happy')  #图名
plt.legend()
plt.show()

num_iterations = 500  #迭代次数
learning_rate = 0.01   #学习率


linear_regression = LinearRegression(x_train,y_train)  #传入训练值，其他默认
(theta,cost_history) = linear_regression.train(learning_rate,num_iterations)   #train函数运算后得到的theta和损失

print ('开始时的损失：',cost_history[0])
print ('训练后的损失：',cost_history[-1])

#损失函数图
plt.plot(range(num_iterations),cost_history)
plt.xlabel('Iter')
plt.ylabel('cost')
plt.title('GD')  #gradient descent=GD+梯度下降
plt.show()

#测试图
predictions_num = 100 #测试数量
x_predictions = np.linspace(x_train.min(),x_train.max(),predictions_num).reshape(predictions_num,1)  #起始位置、终点位置、数量,转矩阵
y_predictions = linear_regression.predict(x_predictions)

plt.scatter(x_train,y_train,label='Train data')
plt.scatter(x_test,y_test,label='test data')
plt.plot(x_predictions,y_predictions,'r',label = 'Prediction')
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.title('Happy')
plt.legend()
plt.show()

# 计算斜率和截距
slope, intercept = np.polyfit(np.linspace(x_train.min(),x_train.max(),predictions_num), y_predictions, 1)

print('Slope:', slope)
print('Intercept:', intercept)
print('拟合回归方程为：y=' , slope , '*x+' , intercept)
Known_y=float(input('Known_y:'))
output_x=(Known_y - intercept) / slope
print('Unknown_x=' , output_x)