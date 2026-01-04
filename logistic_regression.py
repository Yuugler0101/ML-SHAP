import numpy as np
from scipy.optimize import minimize   #command查看
from utils.features import prepare_for_training
from utils.hypothesis import sigmoid
'''scipy数据包用来数理统计分析，引入优化器'''

class LogisticRegression:   #定义类
    #初始化功能
    def __init__(self,data,labels,polynomial_degree = 0,sinusoid_degree = 0,normalize_data=False):  #初始化函数
        """
        1.对数据进行预处理操作
        2.先得到所有的特征个数
        3.初始化参数矩阵
        """
        (data_processed,
         features_mean, 
         features_deviation)  = prepare_for_training(data, polynomial_degree, sinusoid_degree,normalize_data=False)    #画决策边界不能进行归一化处理
        # 通过预处理方程得到的结果（将数据平移除以标准差）：预处理后的数据（加了一列1）、标准化处理后的平均数和偏差

        self.data = data_processed   #date是（150*3）矩阵
        self.labels = labels       #labels是（150*3）矩阵
        self.unique_labels = np.unique(labels)    #计算标签y数量
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data
        # 更新数据x、标签y、标签数量、标准化处理后的平均数、偏差、2个非线性参数、归一化数据

        num_features = self.data.shape[1]   #特征x的数量=数据列的个数（每个含有几个概率值、类别）
        num_unique_labels = np.unique(labels).shape[0]    #标签的数量
        self.theta = np.zeros((num_unique_labels,num_features))   #（需要计算的纬度数、特征值）      #theta是（150*3）矩阵

    # 训练模块功能
    def train(self,max_iterations=1000):  #最大的迭代次数
        cost_histories = []    #损失值记录
        num_features = self.data.shape[1]   #特征个数
        for label_index,unique_label in enumerate(self.unique_labels):  #循环个数等于二分类个数
            current_initial_theta = np.copy(self.theta[label_index].reshape(num_features,1))    #当前标签下初始化theta
            current_labels = (self.labels == unique_label).astype(float)    #当前的标签是否等于循环的标签类别
            (current_theta,cost_history) = LogisticRegression.gradient_descent(self.data,current_labels,current_initial_theta,max_iterations)
            #梯度下降返回值和损失函数，定义梯度下降函数
            self.theta[label_index] = current_theta.T
            cost_histories.append(cost_history)
            
        return self.theta,cost_histories
            
    @staticmethod        #不用实例化类
    #执行预测模块
    def gradient_descent(data,labels,current_initial_theta,max_iterations):  #定义梯度下降函数
        cost_history = []  #返回的损失值
        num_features = data.shape[1]   #特征数量
        result = minimize(
            #要优化的目标：
            #lambda current_theta:LogisticRegression.cost_function(data,labels,current_initial_theta.reshape(num_features,1)),
            lambda current_theta:LogisticRegression.cost_function(data,labels,current_theta.reshape(num_features,1)),
            #初始化的权重参数
            current_initial_theta.flatten(),
            #选择优化策略
            method = 'CG',
            # 梯度下降迭代计算公式
            #jac = lambda current_theta:LogisticRegression.gradient_step(data,labels,current_initial_theta.reshape(num_features,1)),
            jac = lambda current_theta:LogisticRegression.gradient_step(data,labels,current_theta.reshape(num_features,1)),
            # 记录结果
            callback = lambda current_theta:cost_history.append(LogisticRegression.cost_function(data,labels,current_theta.reshape((num_features,1)))),
            # 迭代次数  
            options={'maxiter': max_iterations}                                               
            )   #引用函数
        if not result.success:
            raise ArithmeticError('Can not minimize cost function'+result.message)
        optimized_theta = result.x.reshape(num_features,1)   #最优theta
        return optimized_theta,cost_history
        
    @staticmethod
    #损失计算模块
    def cost_function(data,labels,theat):
        num_examples = data.shape[0]   #总共数据量
        predictions = LogisticRegression.hypothesis(data,theat)    #计算预测值
        y_is_set_cost = np.dot(labels[labels == 1].T,np.log(predictions[labels == 1]))
        #属于当前类别的损失值总和（属于当前类别概率越大，损失值越小）
        y_is_not_set_cost = np.dot(1-labels[labels == 0].T,np.log(1-predictions[labels == 0]))
        #不属于当前类别的损失值总和
        cost = (-1/num_examples)*(y_is_set_cost + y_is_not_set_cost)
        return cost
    @staticmethod
    #预测计算模块
    def hypothesis(data,theat):   #利用函数计算预测值
        predictions = sigmoid(np.dot(data,theat))
        
        return  predictions
    
    @staticmethod
    def gradient_step(data,labels,theta):
        """
                            梯度下降参数更新计算方法，注意是矩阵运算
                """
        num_examples = labels.shape[0]  #样本个数
        predictions = LogisticRegression.hypothesis(data,theta)
        label_diff = predictions- labels   #预测差异
        gradients = (1/num_examples)*np.dot(data.T,label_diff)   #计算梯度
        
        return gradients.T.flatten()  #变平
    
    def predict(self,data):
        """
                            用训练的参数模型，去预测得到概率，根据概率选择类别
                """
        num_examples = data.shape[0]
        data_processed = prepare_for_training(data, self.polynomial_degree, self.sinusoid_degree,self.normalize_data)[0]
        #对数据预处理
        prob = LogisticRegression.hypothesis(data_processed,self.theta.T)
        #date_processed和theta都是（150*3）矩阵
        max_prob_index = np.argmax(prob, axis=1)   #按行索引最大值的位置
        class_prediction = np.empty(max_prob_index.shape,dtype=object)
        for index,label in enumerate(self.unique_labels):   #枚举
            class_prediction[max_prob_index == index] = label   #如果最大值的索引值等哪个标签的索引值，就定义为哪个位置的label
        return class_prediction.reshape((num_examples,1))   #得到类别
        
        
        
        
        
        
        
        
        
        


