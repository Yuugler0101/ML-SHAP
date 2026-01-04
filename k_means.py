import numpy as np

class KMeans:
    def __init__(self,data,num_clustres):   #定义K值
        self.data = data
        self.num_clustres = num_clustres
        
    def train(self,max_iterations):    #迭代次数
        #1.先随机选择K个中心点
        centroids = KMeans.centroids_init(self.data,self.num_clustres)    #质心点初始化
        #2.开始训练
        num_examples = self.data.shape[0]
        closest_centroids_ids = np.empty((num_examples,1))
        for _ in range(max_iterations):
            #3得到当前每一个样本点到K个中心点的距离，找到最近的
            closest_centroids_ids = KMeans.centroids_find_closest(self.data,centroids)
            #4进行中心点位置更新
            centroids = KMeans.centroids_compute(self.data,closest_centroids_ids,self.num_clustres)
        return centroids,closest_centroids_ids
                
    @staticmethod    
    def centroids_init(data,num_clustres):     #质心点初始化（随机选择数据点作为初始聚类中心，并返回这些初始中心的坐标）
        num_examples = data.shape[0]    #数据点的个数
        random_ids = np.random.permutation(num_examples)    #洗牌
        centroids = data[random_ids[:num_clustres],:]     #在洗牌后的数据集中随机选K个点
        return centroids
    @staticmethod 
    def centroids_find_closest(data,centroids):     #3得到当前每一个样本点到K个中心点的距离，找到最近的
        #对于每个数据点，它计算该点与所有聚类中心的距离，并确定最近的聚类中心，将其ID存储在closest_centroids_ids中
        num_examples = data.shape[0]   #样本的个数
        num_centroids = centroids.shape[0]    #中心点的个数
        closest_centroids_ids = np.zeros((num_examples,1))      #存储每个数据点所属的最近的聚类中心的ID
        for example_index in range(num_examples):      #遍历数据点
            distance = np.zeros((num_centroids,1))     #初始化距离集合
            for centroid_index in range(num_centroids):     #遍历中心点
                distance_diff = data[example_index,:] - centroids[centroid_index,:]   #（x1-x0,y1-y0)
                distance[centroid_index] = np.sum(distance_diff**2)    #计算该样本点离质心的距离(平方求和）
            closest_centroids_ids[example_index] = np.argmin(distance)    #找最近的质心，存储ID
        return closest_centroids_ids
    @staticmethod
    def centroids_compute(data,closest_centroids_ids,num_clustres):   #4进行中心点位置更新（首先确定哪些数据点属于每个簇，然后计算每个簇的均值，将其作为新的聚类中心）
        num_features = data.shape[1]
        centroids = np.zeros((num_clustres,num_features))
        for centroid_id in range(num_clustres):    #遍历中心点
            closest_ids = closest_centroids_ids == centroid_id    # 找到属于当前聚类中心的数据点
            centroids[centroid_id] = np.mean(data[closest_ids.flatten(),:],axis=0)    #该类别的平均值
        return centroids
                
            
        
        
        