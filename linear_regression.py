import numpy as np
from utils.features import prepare_for_training

class LinearRegression:

    def __init__(self,data,labels,polynomial_degree = 0,sinusoid_degree = 0,normalize_data=True): #传进数据x（特征）、标签y、2个非线性变换数据、归一化数据
        """
        1.对数据进行预处理操作
        2.先得到所有的特征个数
        3.初始化theta参数矩阵
        """
        (data_processed,
         features_mean, 
         features_deviation)  = prepare_for_training(data, polynomial_degree, sinusoid_degree,normalize_data=True)
        #通过预处理方程得到的结果（将数据平移除以标准差）：预处理后的数据（加了一列1）、标准化处理后的平均数和偏差
         
        self.data = data_processed
        self.labels = labels
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data
        # 更新数据、标签、标准化处理后的平均数、偏差、2个非线性参数、归一化数据

        num_features = self.data.shape[1] #特征x的数量=数据列的个数
        self.theta = np.zeros((num_features,1)) #theta数量等于特征x的个数
        
    def train(self,alpha,num_iterations = 500): #需要迭代次数和学习率
        """
                    训练模块，执行梯度下降
        """
        cost_history = self.gradient_descent(alpha,num_iterations) #迭代后的损失值
        return self.theta,cost_history
        
    def gradient_descent(self,alpha,num_iterations): #需要迭代次数和学习率
        """
                    实际迭代模块，会迭代num_iterations次
        """
        cost_history = []   #记录损失值
        for _ in range(num_iterations):
            self.gradient_step(alpha)    #迭代公式
            cost_history.append(self.cost_function(self.data,self.labels))   #将这一步的损失值记录到损失矩阵中
        return cost_history  #返回损失值
        
        
    def gradient_step(self,alpha):    #进行一次迭代更新，需要学习率
        """
                    梯度下降参数更新计算方法，注意是矩阵运算
        """
        num_examples = self.data.shape[0]     #样本个数
        prediction = LinearRegression.hypothesis(self.data,self.theta)  #得到预测值htheta(x)=当前的theta乘数据x
        delta = prediction - self.labels  #预测值htheta(x)-标签y
        theta = self.theta
        theta = theta - alpha*(1/num_examples)*(np.dot(delta.T,self.data)).T
        '''
                    小批量梯度下降计算公式
        '''
        self.theta = theta
        
        
    def cost_function(self,data,labels):
        """
                    损失计算方法
        """
        num_examples = data.shape[0]  #总的样本个数
        delta = LinearRegression.hypothesis(self.data,self.theta) - labels  #预测值-标签y
        cost = (1/2)*np.dot(delta.T,delta)/num_examples    #均方误差或者其他损失方程
        return cost[0][0]    #返回这一步的损失值
        
        
        
    @staticmethod
    def hypothesis(data,theta):   #训练集：得到预测值htheta(x)=当前的theta乘数据x
        predictions = np.dot(data,theta)
        return predictions   #返回预测值
        
    def get_cost(self,data,labels):  #测试集：计算当前损失
        data_processed = prepare_for_training(data,
         self.polynomial_degree,
         self.sinusoid_degree,
         self.normalize_data
         )[0] #将数据、2个非线性参数、归一化数据预处理，只要date
        
        return self.cost_function(data_processed,labels) #将预处理后的数据和标签带入损失函数，
    def predict(self,data):  #测试集：得到预测值
        """
                    用训练的参数模型，去预测得到回归值结果
        """
        data_processed = prepare_for_training(data,
         self.polynomial_degree,
         self.sinusoid_degree,
         self.normalize_data
         )[0]
         
        predictions = LinearRegression.hypothesis(data_processed,self.theta)
        
        return predictions
