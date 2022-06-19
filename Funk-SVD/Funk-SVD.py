#encoding:utf-8
'''
@author:kiki
@date:2019.03.27
'''

import pickle
import numpy as np
import matplotlib.pyplot as plt


class Funk_SVD(object):
	"""
	implement Funk_SVD
	"""
	def __init__(self, path,USER_NUM,ITEM_NUM,FACTOR):
		super(Funk_SVD, self).__init__()
		self.path = path
		self.USER_NUM=USER_NUM
		self.ITEM_NUM=ITEM_NUM
		self.FACTOR=FACTOR
		self.P=np.random.rand(self.USER_NUM,self.FACTOR)/(self.FACTOR**0.5)
		self.Q=np.random.rand(self.ITEM_NUM,self.FACTOR)/(self.FACTOR**0.5)
		print(self.P.shape)
		print(self.Q.shape)


	def load_data(self,flag='train',sep='\t',random_state=0,size=0.8):
		'''
		flag- train or test
		sep- separator of data
		random_state- seed of the random 
		size- rate of the train of the test
		'''
		np.random.seed(random_state)
		with open(self.path,'r') as f:
			for index,line in enumerate(f):
				if index==0:
					continue
				rand_num=np.random.rand()
				if flag=='train':
					if  rand_num < size:
						u,i,r,t=line.strip('\r\n').split(sep)
						yield (int(u)-1,int(i)-1,float(r))
				else:
					if rand_num >= size:
						u,i,r,t=line.strip('\r\n').split(sep)
						yield (int(u)-1,int(i)-1,float(r))
	
	def train(self,epochs=5,theta=1e-4,alpha=0.02,beta=0.02):#500
		'''
		train the model
		epochs- num of iterations
		theta- therehold of iterations
		alpha- learning rate
		beta- parameter of regularization term
		'''
		old_e=0.0
		self.cost_of_epoch=[]
		for epoch in range(epochs):#SGD
			print("current epoch is {}".format(epoch))
			current_e=0.0
			train_data=self.load_data(flag='train') #reload the train data every iteration(generator)
			for index,d in enumerate(train_data): 
				u,i,r=d
				pr=np.dot(self.P[u],self.Q[i])
				err=r-pr 
				current_e+=pow(err,2) #loss term
				self.P[u]+=alpha*(err*self.Q[i]-beta*self.P[u])
				self.Q[i]+=alpha*(err*self.P[u]-beta*self.Q[i])
				current_e+=(beta/2)*(sum(pow(self.P[u],2))+sum(pow(self.Q[i],2))) #正则项
			self.cost_of_epoch.append(current_e)
			print('cost is {}'.format(current_e))
			if abs(current_e - old_e) < theta:
				break
			old_e=current_e
			alpha*=0.9


	def predict_rating(self,user_id,item_id):
		'''
		predict rating for target user of target item

		user- the number of user(user_id=xuhao-1)
		item- the number of item(item_id=xuhao-1)
		'''
		pr=np.dot(self.P[user_id],self.Q[item_id])
		return pr

	def recommand_list(self,user,k=10):
		'''
		recommand top n for target user
		for rating prediction,recommand the items which socre is higer than 4/5 of max socre
		'''
		user_id=user-1
		user_items={}
		for item_id in range(self.ITEM_NUM):
			user_had_look = {}
			user_had_look = self.user_had_look_in_train()
			if item_id in user_had_look[user]:
			   continue
			pr=self.predict_rating(user_id,item_id)
			user_items[item_id]=pr
		items=sorted(user_items.items(),key=lambda x:x[1],reverse=True)[:k]
		return items
    
	def user_had_look_in_train(self):
		user_had_look = {}
		train_data=self.load_data(flag='train')
		for index,d in enumerate(train_data):
			u,i,r=d
			user_had_look.setdefault(u,{})
			user_had_look[u][i] = r
		return user_had_look


	def test_rmse(self):
		'''
		test the model and return the value of rmse
		'''
		rmse=.0
		num=0
		test_data=self.load_data(flag='test')
		for index,d in enumerate(test_data):
			num=index+1
			u,i,r=d
			pr=np.dot(self.P[u],self.Q[i])
			rmse+=pow((r-pr),2)
		rmse=(rmse/num)**0.5
		return rmse
	
	def show(self):
		'''
		show figure for cost and epoch
		'''
		nums=range(len(self.cost_of_epoch))
		plt.plot(nums,self.cost_of_epoch,label='cost value')
		plt.xlabel('# of epoch')
		plt.ylabel('cost')
		plt.legend()
		plt.show()
		pass

	def save_model(self):
		'''
		save the model to pickle,P,Q and rmse
		'''
		data_dict={'P':self.P,'Q':self.Q}
		f=open('funk-svd.pkl','wb')
		pickle.dump(data_dict,f)
		pass

	def read_model(self):
		'''
		reload the model from local disk
		'''
		f=open('funk-svd.pkl','rb')
		model=pickle.load(f)
		self.P=model['P']
		self.Q=model['Q']
	
	def evaluate(self):
		print('Evaluating start ...')
		movie_popular = {}
		for user, movies in enumerate(self.load_data(flag='train')):#一名用户下，对部分电影的评分列表
			for movie in movies:
				if movie not in movie_popular:
					movie_popular[movie] = 0
				movie_popular[movie] += 1 #统计电影出现的次数？因为一些电影没有用户看过，
                                                #同时如果次数很多，说明很多用户看过，那它就算更流行，则需要减少流行电影（热门电影）对推荐的影响
		movie_count = len(movie_popular)

		N = 10
		# 准确率和召回率
		hit = 0
		rec_count = 0
		test_count = 0
		# 覆盖率
		all_rec_movies = set()

		for i, user in enumerate(self.load_data(flag='train')):
			test_moives = self.load_data(flag='test')
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
		if movie_count!=0:
			coverage = len(all_rec_movies) / (1.0 * movie_count)
		else:
			coverage = 0
		# print('precisioin=%.4f\trecall=%.4f\tcoverage=%.4f' % (precision, recall, coverage))
		print('precisioin=%.2f%%\t recall=%.2f%%\t coverage=%.2f%%' % (precision*100, recall*100, coverage*100))

if __name__=="__main__":
	mf=Funk_SVD(r'ml-100k/u.data',943,1682,50)#path,user_num,item_num,factor(向量纬度)
	# mf.train()
	# # mf.save_model()
	# rmse=mf.test_rmse()
	# print("rmse:",rmse)
	# user_items=mf.recommand_list(3)
	# print(user_items)
	# mf.evaluate()
	for i, user in enumerate(mf.load_data(flag='train')):
		print(i, user, user[0])
		test_moives = list(mf.load_data(flag='test'))
		print(len(test_moives))
		# rec_movies = self.recommend(user)
		# rec_movies = self.recommand_list(user)
		# for movie, w in rec_movies:
		# 	if movie in test_moives:
		# 		hit += 1
		# 	all_rec_movies.add(movie)
		# rec_count += N
		# test_count += len(test_moives)