# -*- coding: UTF-8 -*-
import operator
import pickle
from math import log
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def createDataSet():#创建数据集
	dataSet = [[0, 0, 0, 0, 'no'],						
			[0, 0, 0, 1, 'no'],
			[0, 1, 0, 1, 'yes'],
			[0, 1, 1, 0, 'yes'],
			[0, 0, 0, 0, 'no'],
			[1, 0, 0, 0, 'no'],
			[1, 0, 0, 1, 'no'],
			[1, 1, 1, 1, 'yes'],
			[1, 0, 1, 2, 'yes'],
			[1, 0, 1, 2, 'yes'],
			[2, 0, 1, 2, 'yes'],
			[2, 0, 1, 1, 'yes'],
			[2, 1, 0, 1, 'yes'],
			[2, 1, 0, 2, 'yes'],
			[2, 0, 0, 0, 'no']]     #四个特征，一个标签
	labels = ['F1-AGE', 'F2-WORK', 'F3-HOME', 'F4-LOAN']		
	return dataSet, labels


def createTree(dataset,labels,featLabels):   #创建树模型（数据集、标签、按照选择顺序对特征进行排序）
	classList = [example[-1] for example in dataset]    #是否需要继续划分节点=判断叶子节点标签是否都一样
	# （example[-1]：最后一列）
	if classList.count(classList[0]) == len(classList):
		return classList[0]
	if len(dataset[0]) == 1:
		return majorityCnt(classList)  #当分类到只剩标签，遍历结束（之后根据返回值，计算类别的重数）
	bestFeat = chooseBestFeatureToSplit(dataset)  #选择最优的特征
	bestFeatLabel = labels[bestFeat]     #找到最优特征的标签
	featLabels.append(bestFeatLabel)    #传进标签
	myTree = {bestFeatLabel:{}}     #字典里嵌套字典
	del labels[bestFeat]    #删掉最优的列
	featValue = [example[bestFeat] for example in dataset]   #统计某一特征有几个属性值
	uniqueVals = set(featValue)   #set函数：无序且不含重复元素
	for value in uniqueVals:   #继续选择节点
		sublabels = labels[:]   #切分后的数据集
		myTree[bestFeatLabel][value] = createTree(splitDataSet(dataset,bestFeat,value),sublabels,featLabels)
		#新的树模型
	return myTree

def majorityCnt(classList):   #计算类别的重数
	classCount={}  #计数
	for vote in classList:   #遍历classList寻找个数多的
		if vote not in classCount.keys():classCount[vote] = 0
		#classCount.keys()“”“遍历classCount的所有属性”“”
		#:classCount[vote]=0“”“未存在过的值则初始化等于0”“”
		classCount[vote] += 1
	sortedclassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
	#排序，返回出现次数最多的
	return sortedclassCount[0][0]

def chooseBestFeatureToSplit(dataset):   #选择最好的特征
	numFeatures = len(dataset[0]) - 1     #当前特征个数=特征数-1
	baseEntropy = calcShannonEnt(dataset)   #基础熵值计算
	bestInfoGain = 0
	bestFeature = -1
	for i in range(numFeatures):         #遍历特征
		featList = [example[i] for example in dataset]     #当前这一列的特征
		uniqueVals = set(featList)       #特征有几个类型
		newEntropy = 0      #初始化熵值
		for val in uniqueVals:    #计算每个特征的熵值，遍历每个类型
			subDataSet = splitDataSet(dataset,i,val)    #切分数据集
			prob = len(subDataSet)/float(len(dataset))   #计算概率值
			newEntropy += prob * calcShannonEnt(subDataSet)    #计算熵值和
		infoGain = baseEntropy - newEntropy    #信息增益
		if (infoGain > bestInfoGain):   #如果信息增益值大于当前最优的信息增益，则定义为新的最优值和信息增益
			bestInfoGain = infoGain
			bestFeature = i	
	return bestFeature
			
			
			
			
			
		
def splitDataSet(dataset,axis,val):   #切分数据集
	retDataSet = []   #子集
	for featVec in dataset:   #遍历每一个特征
		if featVec[axis] == val:   #是否等于该特征值的某一个类型
			reducedFeatVec = featVec[:axis]
			reducedFeatVec.extend(featVec[axis+1:])   #把当前这一列去掉
			retDataSet.append(reducedFeatVec)
	return retDataSet
			
def calcShannonEnt(dataset):   #计算熵值
	numexamples = len(dataset)   #所有的样本组个数
	labelCounts = {}    #计算各个标签的个数
	for featVec in dataset:
		currentlabel = featVec[-1]   #提取标签列
		if currentlabel not in labelCounts.keys():    #如果当前的标签值不在字典里
			labelCounts[currentlabel] = 0   #初始化结果为0
		labelCounts[currentlabel] += 1

	shannonEnt = 0     #初始化熵值为0
	for key in labelCounts:     #对每一个类别进行遍历
		prop = float(labelCounts[key])/numexamples    #概率值
		shannonEnt -= prop*log(prop,2)       #计算熵值和
	return shannonEnt

'''绘图操作'''
def getNumLeafs(myTree):#获取决策树叶结点数目
	numLeafs = 0												
	firstStr = next(iter(myTree))				#iter:迭代器；next：下一个
	secondDict = myTree[firstStr]								
	for key in secondDict.keys():
	    if type(secondDict[key]).__name__=='dict':				
	        numLeafs += getNumLeafs(secondDict[key])
	    else:   numLeafs +=1
	return numLeafs


def getTreeDepth(myTree):#获取决策树层数
	maxDepth = 0												
	firstStr = next(iter(myTree))								
	secondDict = myTree[firstStr]								
	for key in secondDict.keys():
	    if type(secondDict[key]).__name__=='dict':				
	        thisDepth = 1 + getTreeDepth(secondDict[key])
	    else:   thisDepth = 1
	    if thisDepth > maxDepth: maxDepth = thisDepth			
	return maxDepth

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
	arrow_args = dict(arrowstyle="<-")											
	font = FontProperties(fname=r"c:\windows\fonts\simsunb.ttf", size=14)		
	createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',	
		xytext=centerPt, textcoords='axes fraction',
		va="center", ha="center", bbox=nodeType, arrowprops=arrow_args, FontProperties=font)


def plotMidText(cntrPt, parentPt, txtString):
	xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]																
	yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
	createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


def plotTree(myTree, parentPt, nodeTxt):
	decisionNode = dict(boxstyle="sawtooth", fc="0.8")										
	leafNode = dict(boxstyle="round4", fc="0.8")											
	numLeafs = getNumLeafs(myTree)  														
	depth = getTreeDepth(myTree)															
	firstStr = next(iter(myTree))																								
	cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)	
	plotMidText(cntrPt, parentPt, nodeTxt)													
	plotNode(firstStr, cntrPt, parentPt, decisionNode)										
	secondDict = myTree[firstStr]															
	plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD										
	for key in secondDict.keys():								
		if type(secondDict[key]).__name__=='dict':											
			plotTree(secondDict[key],cntrPt,str(key))        								
		else:																														
			plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
			plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
			plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
	plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD


def createPlot(inTree):
	fig = plt.figure(1, facecolor='white')													#创建fig
	fig.clf()																				#清空fig
	axprops = dict(xticks=[], yticks=[])
	createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)    							#去掉x、y轴
	plotTree.totalW = float(getNumLeafs(inTree))											#获取决策树叶结点数目
	plotTree.totalD = float(getTreeDepth(inTree))											#获取决策树层数
	plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;								#x偏移
	plotTree(inTree, (0.5,1.0), '')															#绘制决策树
	plt.show()




if __name__ == '__main__':
	dataset, labels = createDataSet()   #返回值
	featLabels = []
	myTree = createTree(dataset,labels,featLabels)
	createPlot(myTree)
	
	
	
	
	

	
	






						
