#encoding:utf-8
'''
@author:kiki
@date:2019.03.27
'''

import pickle
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import json
from operator import itemgetter


class Funk_SVD(object):
	"""
	implement Funk_SVD
	"""
	def __init__(self, path,USER_NUM,ITEM_NUM,FACTOR):
		super(Funk_SVD, self).__init__()
		self.trainSet = {}
		self.testSet = {}
		self.u = set() #user
		self.m = set() #movie
		self.movie_count = 0


		self.path = path
		self.USER_NUM = USER_NUM+1
		self.ITEM_NUM = ITEM_NUM+1
		self.FACTOR = FACTOR
		#初始化分解矩阵
		self.P = np.random.rand(self.USER_NUM,self.FACTOR)/(self.FACTOR**0.5)
		self.Q = np.random.rand(self.ITEM_NUM,self.FACTOR)/(self.FACTOR**0.5)

		print(self.P.shape)
		print(self.Q.shape)
		# self.trainSet_u = set()
		# self.trainSet_i = set()
		self.trainSet_len = 0
	# 读文件，返回文件的每一行
	def load_file(self, filename):
		with open(filename, 'r') as f:
			for i, line in enumerate(f):
				if i == 0:  # 去掉文件第一行的title
					continue
				yield line.strip('\r\n')
		print('Load %s success!' % filename)
	def get_dataSet(self, filename, pivot=0.8):
		trainSet_len = 0
		testSet_len = 0
		if os.path.exists("trainSet1.json"):
			# pass
			with open("trainSet1.json", "r", encoding='GBK') as f:
				self.trainSet = json.load(f)
				f.close()
			with open("testSet1.json", "r") as f:
				self.testSet = json.load(f)
				f.close()
		else:
			for line in self.load_file(filename):
				user, movie, rating, timestamp = line.split('\t')
				if(random.random() < pivot):
					self.trainSet.setdefault(user, {})
					self.trainSet[user][movie] = rating
					self.u.add(user)
					self.m.add(movie)
					trainSet_len += 1
				else:
					self.testSet.setdefault(user, {})
					self.testSet[user][movie] = rating
					testSet_len += 1
			print('Split trainingSet and testSet success!')
			with open('trainSet1.json', 'w') as f:
				json.dump(self.trainSet, f)   #data转换为json数据格式并写入文件
				f.close()  #关闭文件
			with open('testSet1.json', 'w') as f:
				json.dump(self.testSet, f)   #data转换为json数据格式并写入文件
				f.close()  #关闭文件
			print('TrainSet = %s' % trainSet_len)
			print('TestSet = %s' % testSet_len)

	# def load_data(self,flag = 'train',sep = '\t',random_state = 0,size = 0.8):
	# 	'''
	# 	flag- train or test
	# 	sep- separator of data
	# 	random_state- seed of the random 
	# 	size- rate of the train of the test（划分训练集和测试集）
	# 	'''
	# 	np.random.seed(random_state)
	# 	with open(self.path,'r') as f:
	# 		for index,line in enumerate(f):
	# 			if index == 0:#跳过第一行（标题行）
	# 				continue
	# 			rand_num = np.random.rand()
	# 			if flag == 'train':
	# 				if  rand_num < size:
	# 					u,i,r,t = line.strip('\r\n').split(sep)
	# 					# self.trainSet_u.add(u)
	# 					# self.trainSet_i.add(i)
	# 					self.trainSet_len = self.trainSet_len+1
	# 					yield (int(u)-1,int(i)-1,float(r))
	# 			else:
	# 				if rand_num >= size:
	# 					u,i,r,t = line.strip('\r\n').split(sep)
	# 					yield (int(u)-1,int(i)-1,float(r))
	
	def train(self,epochs = 5,theta = 1e-4,alpha = 0.002,beta = 0.02):#500
		'''
		train the model
		epochs- num of iterations  （迭代次数）
		theta- therehold of iterations
		alpha- learning rate
		beta- parameter of regularization term
		'''
		old_e = 0.0
		self.cost_of_epoch = []
		all_data = []
		for epoch in range(epochs):#SGD
			print("current epoch is {}".format(epoch))
			current_e = 0.0
			# train_data = self.load_data(flag = 'train') #reload the train data every iteration(generator)
			# print(train_data)
			# train_data = self.load_data(flag = 'train',sep = ',')
			# print(self.P)
			for u, i in self.trainSet.items(): 
				# u,i,r = d[]
				# print(u)
				# print(i)
				u = int(u)
				# print(i)
				# print(index, u,i,r)
				# all_data.append([index, u, i ,r])
				for movie in i:
					r = float(i[movie])
					movie = int(movie)
					
					# print(self.P[int(u)])
					# print(self.Q[int(movie)].shape)
					pr = np.dot(self.P[u], self.Q[movie])
					err = r-pr 
					current_e += pow(err,2) #loss term
					self.P[u] += alpha*(err*self.Q[movie]-beta*self.P[u])
					self.Q[movie] += alpha*(err*self.P[u]-beta*self.Q[movie])
					current_e += (beta/2)*(sum(pow(self.P[u],2))+sum(pow(self.Q[movie],2))) #正则项
			# print("self.trainSet_u = ",len(self.trainSet_u))
			# print("self.trainSet_i = ",len(self.trainSet_i))
			self.cost_of_epoch.append(current_e)
			print('cost is {}'.format(current_e))
			if abs(current_e - old_e) < theta:
				break
			old_e = current_e
			alpha *= 0.9

		# print("all_data = ",all_data)
		

	def predict_rating(self,user_id,item_id):
		'''
		predict rating for target user of target item

		user- the number of user(user_id = xuhao-1)
		item- the number of item(item_id = xuhao-1)
		'''
		pr = np.dot(self.P[int(user_id)],self.Q[int(item_id)])
		return pr

	def recommand_list(self,user,k = 10):
		'''
		recommand top n for target user
		for rating prediction,recommand the items which socre is higer than 4/5 of max socre
		'''
		user_id = user
		user_items = {}
		user_had_look = self.trainSet
		for item_id in range(self.ITEM_NUM):
			if item_id in user_had_look[str(user)]:
			   continue
			pr = self.predict_rating(user_id, item_id)
			user_items[item_id] = pr
		items = sorted(user_items.items(),key = lambda x:x[1],reverse = True)[:k]
		return items
    
	# def user_had_look_in_train(self):
	# 	# print("user_had_look_in_train")
	# 	user_had_look = {}
	# 	# train_data = self.load_data(flag = 'train')
	# 	# train_data = self.load_data(flag = 'train', sep = ',')
	# 	# print("load_data end")
	# 	with open("trainSet1.json", "r", encoding='GBK') as f:
	# 		self.trainSet = json.load(f)
	# 		f.close()

	# 	# for index,d in enumerate(train_data):
	# 	# 	u,i,r = d
	# 	# 	user_had_look.setdefault(u,{})
	# 	# 	user_had_look[u][i] = r
	# 	# return user_had_look
	# 	return self.trainSet


	def test_rmse(self):
		'''
		test the model and return the value of rmse（均值方差，越小越好）
		'''
		rmse = .0
		# num = 0
		num = 1
		with open("testSet1.json", "r") as f:
			self.testSet = json.load(f)
			f.close()
		# print(self.testSet)
		# test_data = self.load_data(flag = 'test')
		# test_data = self.load_data(flag = 'test', sep = ',')
		# print(test_data)
		for u, i in self.testSet.items(): 
			u = int(u)
			for movie in i:
				r = float(i[movie])
				movie = int(movie)
				pr = np.dot(self.P[u],self.Q[movie])
				rmse += pow((r-pr),2)
			rmse = (rmse/num)**0.5
			num = num+1

		# for index,d in enumerate(test_data):
		# 	# print(index,d)
		# 	num = index+1
		# 	u,i,r = d
		# 	pr = np.dot(self.P[u],self.Q[i])
		# 	rmse += pow((r-pr),2)
		# rmse = (rmse/num)**0.5
		return rmse
	
	def show(self):
		'''
		show figure for cost and epoch
		'''
		# nums = range(len(self.cost_of_epoch))
		# plt.plot(nums,self.cost_of_epoch,label = 'cost value')
		# plt.xlabel('# of epoch')
		# plt.ylabel('cost')
		# plt.legend()
		# plt.show()
		pass

	def save_model(self):
		'''
		save the model to pickle,P,Q and rmse
		'''
		data_dict = {'P':self.P,'Q':self.Q}
		f = open('funk-svd.pkl','wb')
		pickle.dump(data_dict,f)
		pass

	def read_model(self):
		'''
		reload the model from local disk
		'''
		f = open('funk-svd.pkl','rb')
		model = pickle.load(f)
		self.P = model['P']
		self.Q = model['Q']
	
	def evaluate(self):
		print('Evaluating start ...')
		movie_popular = {}
		for user, movies in self.trainSet.items():#一名用户下，对部分电影的评分列表
			for movie in movies:
				if movie not in movie_popular:
					movie_popular[movie] = 0
				movie_popular[movie] += 1 #统计电影出现的次数？因为一些电影没有用户看过，
                                                #同时如果次数很多，说明很多用户看过，那它就算更流行，则需要减少流行电影（热门电影）对推荐的影响
		self.movie_count = len(movie_popular)

		N = 10
		# 准确率和召回率
		hit = 0
		rec_count = 0
		test_count = 0
		# 覆盖率
		all_rec_movies = set()

		for i, user in enumerate(self.trainSet):
			test_moives = self.testSet.get(user, {})
			# rec_movies = self.recommend(user)
			rec_movies = self.recommand_list(user)
			for movie, w in rec_movies:
				if movie in test_moives:
					hit += 1
				all_rec_movies.add(movie)
			rec_count += N
			test_count += len(test_moives)

		precision = hit / (1.0 * rec_count)
		recall = hit / (1.0 * test_count)
		if self.movie_count!=0:
			coverage = len(all_rec_movies) / (1.0 * self.movie_count)
		else:
			coverage = 0
		# print('precisioin=%.4f\trecall=%.4f\tcoverage=%.4f' % (precision, recall, coverage))
		print('precisioin=%.2f%%\t recall=%.2f%%\t coverage=%.2f%%' % (precision*100, recall*100, coverage*100))

if __name__ == "__main__":
	# path, user_num, item_num, factor
	mf = Funk_SVD(r'',943,1682,50)
	mf.get_dataSet("ml-100k/u.data")
	#path, user_num, item_num, factor
	# mf = Funk_SVD(r'../data/rand_set7.csv',1371,125916,50)
	# mf.train()
	# mf.save_model()
	mf.read_model() #加载P、Q
	rmse = mf.test_rmse() #计算均值方差大小
	print("rmse:",rmse)
	user_items = mf.recommand_list(3)#给用户3推荐电影
	print(user_items)
	mf.evaluate()


