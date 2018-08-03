#coding:utf-8
import math
import copy
import numpy as np
import matplotlib.pyplot as plt

isdebug = True

#指定k个高斯分布参数，这里指定k=2。
#注意2个高斯分布具有相同方差Sigma，均值分别为Mu1,Mu2。
#共1000个数据


#Mu1,Mu2两个分布的均值，k分布个数
#N：每个分布下的总的样本数
#Sigma:方差，
def ini_data(Sigma,Mu1,Mu2,k,N):
  global X #保存生成的随机样本
  global Mu #求类别的均值
  global Expectations #保存样本属于某类的概率
  
  #1*N的矩阵，生成N个样本
  X = np.zeros((1,N)) 
  #初始均值：两个分布的均值
  Mu = np.random.random(2) #0-1之间
  print(Mu)
  #给定1000*2的矩阵，保存样本属于某类的概率
  Expectations = np.zeros((N,k)) 
  #生成N个样本数据 
  for i in range(0,N): 
    #在大于0.5在第1个分布，小于0.5在第2个分布
    if np.random.random(1) > 0.5:
	  #N(Mu1,Sigma)的正态分布，#标准正态分布np.random.normal()
      X[0,i] = np.random.normal()*Sigma + Mu1 #
    else:
	  #N(Mu2,Sigma)正态分布
      X[0,i] = np.random.normal()*Sigma + Mu2 
	  
  if isdebug:
    print("***********")
    print(u"初始观测数据X：")
    print(X)
    
	
#E步 计算每个样本属于男女各自的概率
#输入：方差Sigma，类别k，样本数N
def e_step(Sigma,k,N):
  #样本属于某类概率
  global Expectations
  #两类均值
  global Mu
  #样本
  global X

  #遍历所有样本点，计算属于每个类别的概率
  for i in range(0,N):
    #分母，用于归一化
    Denom = 0
	#遍历男女两类，计算各自归一化分母
    for j in range(0,k):
	  #计算分母
      Denom += math.exp((-1/(2*(float(Sigma**2))))*(float(X[0,i]-Mu[j]))**2)

	#遍历男女两类，计算各自分子部分
    for j in range(0,k):
	  #分子
      Numer = math.exp((-1/(2*(float(Sigma**2))))*(float(X[0,i]-Mu[j]))**2)
      #每个样本属于该类别的概率
      Expectations[i,j] = Numer/Denom

  if isdebug:
    print("***********")
    print(u"隐藏变量E（Z）：")
    print(len(Expectations))
	#数据总个数
    print(Expectations.size)
	#矩阵数据
    print(Expectations.shape)
	#打印出隐藏变量的值
    print(Expectations)


#M步 期望最大化
def m_step(k,N):
  #样本属于某类概率P(k|xi)
  global Expectations
  #样本
  global X
  #计算两类的均值
  #遍历两类
  for j in range(0,k):
    Numer = 0
    Denom = 0
	#当前类别下，遍历所有样本
	#计算该类别下的均值和方差
    for i in range(0,N):
	  #该类别样本分布P(k|xi)xi
      Numer += Expectations[i,j]*X[0,i]
	  #该类别类样本总数Nk，Nk等于P(k|xi)求和
      Denom +=Expectations[i,j]
	#计算每个类别各自均值uk
    Mu[j] = Numer / Denom


#算法迭代iter_num次，或达到精度Epsilon停止迭代
#迭代次数1000次， 误差达到0.0001终止
#输入：两类相同方差Sigma，一类均值Mu1，一类均值Mu2
#类别数k，样本数N，迭代次数iter_num，可接受精度Epsilon
def run(Sigma,Mu1,Mu2,k,N,iter_num,Epsilon):
  #生成训练样本
  ini_data(Sigma,Mu1,Mu2,k,N)
  print(u"初始<u1,u2>:", Mu)

  #迭代1000次
  for i in range(iter_num):
    #保存上次两类均值
    Old_Mu = copy.deepcopy(Mu)
	#E步
    e_step(Sigma,k,N)
    #M步
    m_step(k,N)

    #输出当前迭代次数及当前估计的值
    print(i,Mu)

	#判断误差
    if sum(abs(Mu-Old_Mu)) < Epsilon:
      break
	  
if __name__ == '__main__':

   #随机生成两个分布的数据集
   ini_data(6,40,20,2,1000)
   plt.hist(X[0,:],100) #柱状个数（纵轴表示区间段样本数）
   plt.show()

  ##混合的两个高斯分布数据，估计各个分布的均值
  ##sigma,mu1,mu2,模型数，样本总数，迭代次数，迭代终止收敛精度
   run(6,40,20,2,1000,1000,0.0001)
   plt.hist(X[0,:],100) #柱状图的宽度
   plt.show()

   

