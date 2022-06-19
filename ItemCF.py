# coding = utf-8

# 基于项目的协同过滤推荐算法实现
import random
import math
import os

from operator import itemgetter
import numpy as np
import json

class ItemBasedCF():
    # 初始化参数
    def __init__(self):
        # 找到相似的20部电影，为目标用户推荐10部电影
        self.n_sim_movie = 20
        self.n_rec_movie = 10

        # 将数据集划分为训练集和测试集
        self.trainSet = {}
        self.testSet = {}

        # 用户相似度矩阵（算法）
        self.movie_sim_matrix = {}
        self.movie_popular = {}
        self.movie_count = 0

        #矩阵分解
        self.u = set() #user
        self.m = set() #movie
        self.user_movie_maxtri = {} #原始的用户评分矩阵
        self.user = tuple(self.u)
        self.movie = tuple(self.m)
        self.all_score = {}  #得到所有实际评分+预测评分

        print('Similar movie number = %d' % self.n_sim_movie)
        print('Recommneded movie number = %d' % self.n_rec_movie)


    # 读文件得到“用户-电影”数据======》用户对每部电影的评分
    #结构
    # user1:{
    #     movie1:score1,
    #     movie2:score2
    # }
    # user2:{
    #     movie1:score1,
    #     movie2:score2
    # }
    def get_dataset(self, filename, pivot=0.75):
        trainSet_len = 0
        testSet_len = 0
        for line in self.load_file(filename):
            user, movie, rating, timestamp = line.split(',')
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
        with open('train/trainSet'+str(trainSet_len)+'.json', 'w') as f:
            json.dump(self.trainSet, f)   #data转换为json数据格式并写入文件
            f.close()  #关闭文件
        with open('train/testSet'+str(testSet_len)+'.json', 'w') as f:
            json.dump(self.testSet, f)   #data转换为json数据格式并写入文件
            f.close()  #关闭文件
        print('TrainSet = %s' % trainSet_len)
        print('TestSet = %s' % testSet_len)


    # 读文件，返回文件的每一行
    def load_file(self, filename):
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                if i == 0:  # 去掉文件第一行的title
                    continue
                yield line.strip('\r\n')
        print('Load %s success!' % filename)


    # 计算电影之间的相似度
    def calc_movie_sim(self):
        for user, movies in self.trainSet.items():#一名用户下，对部分电影的评分列表
            for movie in movies:
                if movie not in self.movie_popular:
                    self.movie_popular[movie] = 0
                self.movie_popular[movie] += 1 #统计电影出现的次数？因为一些电影没有用户看过，
                                                #同时如果次数很多，说明很多用户看过，那它就算更流行，则需要减少流行电影（热门电影）对推荐的影响


        self.movie_count = len(self.movie_popular)
        print("Total movie number = %d" % self.movie_count)
         # 用户相似度矩阵（算法）
        for user, movies in self.trainSet.items():
            for m1 in movies:
                for m2 in movies:
                    if m1 == m2:
                        continue
                    # print("user, m1, m2", (user, m1, m2))
                    self.movie_sim_matrix.setdefault(m1, {})
                    self.movie_sim_matrix[m1].setdefault(m2, 0)
                    self.movie_sim_matrix[m1][m2] += 1
        # with open("./movie_sim_matrix.txt", 'w+') as f:
            # f.write(str(self.movie_sim_matrix))
        print("Build co-rated users matrix success!")

        # 计算电影之间的相似性
        print("Calculating movie similarity matrix ...")
        for m1, related_movies in self.movie_sim_matrix.items():
            # m1  related_movies  2 {'29': 10, '32': 64, '47': 72, '50': 60, '112': 20, '151': 33, 
            # print("m1  related_movies ", m1,related_movies)
            for m2, count in related_movies.items():
                # 注意0向量的处理，即某电影的用户数为0
                if self.movie_popular[m1] == 0 or self.movie_popular[m2] == 0:
                    self.movie_sim_matrix[m1][m2] = 0
                else:
                    #计算用户的相似度
                    self.movie_sim_matrix[m1][m2] = count / math.sqrt(self.movie_popular[m1] * self.movie_popular[m2])
                    #self.movie_sim_matrix[m1][m2] = count / math.sqrt((self.movie_popular[m1] * self.movie_popular[m1])+(self.movie_popular[m2]*self.movie_popular[m2]))#itemCF针对w
                    #self.movie_sim_matrix[m1][m2] = count / (math.sqrt(self.movie_popular[m1] * self.movie_popular[m1])*(math.sqrt(self.movie_popular[m2] * self.movie_popular[m2])))
        print('Calculate movie similarity matrix success!')


    # 针对目标用户U，找到K部相似的电影，并推荐其N部电影（这是关键的推荐算法）
    #定义Auser为被推荐者
    def recommend(self, user):
        K = self.n_sim_movie  #相似的电影数目
        N = self.n_rec_movie    #要推荐的电影数目
        rank = {}
        watched_movies = self.trainSet[user]    #取出Auser喜欢的电影list
        # print("watched_movies={:}", format(watched_movies))
        for movie, rating in watched_movies.items():    #movie_sim_matrix 每一部Auser喜欢的电影，其相似电影有哪些
            # print("movie_sim_matrix[movie] = {:}",format(self.movie_sim_matrix[movie]))
            #related_movie表示相关的电影ID，w表示相似权重值大小
            for related_movie, w in sorted(self.movie_sim_matrix[movie].items(), key=itemgetter(1), reverse=True)[:K]:
                if related_movie in watched_movies: #推荐的电影中有Auser喜欢的,那就跳过
                    continue
                rank.setdefault(related_movie, 0)   #w是什么（应该类似于权重的东西，跟评分相乘之后会得到Auser可能的评分），很关键
                rank[related_movie] += w * float(rating)#相似度*评分  就可以得到用户可能喜欢的电影
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[:N]    #返回排序后的推荐的电影

    #使用矩阵分解法
    def recommand_maxtri(self):
        # print("self.trainSet", self.trainSet)
        self.user_movie_maxtri = np.zeros((len(self.u), len(self.m)), dtype=float)
        self.user = tuple(self.u)
        self.movie = tuple(self.m)
        print("user  len ",len(self.user))
        print("movie  len ",len(self.movie))
        with open("train/user.txt", "w") as f:
            f.write(str(self.user))
            f.close()
        with open("train/movie.txt", "w") as f:
            f.write(str(self.movie))
            f.close()
        print("构建user_movie_maxtri")
        jishu = 1
        for us in self.user:
            for mov in self.movie:
                # print(user.index(us), movie.index(mov), self.trainSet[us][mov])
                if us in self.trainSet.keys() and mov in self.trainSet[us].keys():
                    self.user_movie_maxtri[self.user.index(us)][self.movie.index(mov)] = float(self.trainSet[us][mov])
                else:
                    self.user_movie_maxtri[self.user.index(us)][self.movie.index(mov)] = 0
        self.user_movie_maxtri.tofile("train/user_movie_maxtri.bin")
        # print("user_movie_maxtri", self.user_movie_maxtri)
        # print("p", p)
        # print("q", q)
        loss_list = [] #存储每次迭代计算的loss值
        N = len(self.user)
        M = len(self.movie)
        dicts = {
            "N":N,
            "M":M
        }
        with open("train/shape_num.json", "w") as f:
            json.dump(dicts, f)
            f.close()
        K = N
        alpha=0.0002
        beta=0.02
        P = np.random.rand(N,K) #
        Q = np.random.rand(M,K) #
        Q = Q.T #Q矩阵转置
        print("开始矩阵分解")
        for step in range(5000):
            #更新R^
            for i in range(N):
                for j in range(M):
                    if self.user_movie_maxtri[i][j] != 0:
                        #计算损失函数
                        error = self.user_movie_maxtri[i][j]
                        for k in range(K):
                            error -= P[i][k]*Q[k][j]
                        #优化P,Q矩阵的元素
                        for k in range(K):
                            P[i][k] = P[i][k] + alpha*(2*error*Q[k][j]-beta*P[i][k])
                            Q[k][j] = Q[k][j] + alpha*(2*error*P[i][k]-beta*Q[k][j])
            
            loss = 0.0
            #计算每一次迭代后的loss大小，就是原来R矩阵里面每个非缺失值跟预测值的平方损失
            for i in range(N):
                for j in range(M):
                    if self.user_movie_maxtri[i][j] != 0:
                        #计算loss公式加号的左边
                        data = 0
                        for k in range(K):
                            data = data + P[i][k]*Q[k][j]
                        loss = loss + math.pow(self.user_movie_maxtri[i][j]-data,2)
                        #得到完整loss值
                        for k in range(K):
                            loss = loss + beta/2*(P[i][k]*P[i][k]+Q[k][j]*Q[k][j])
                        # loss_list.append(loss)
            #输出loss值
            if (step+1) % 1000 == 0:
                print("loss={:}".format(loss))
                P.tofile("P.bin")
                Q.tofile("Q.bin")
            #判断
            if loss < 0.001:
                print(loss)
                break
        # print(loss_list)
        print("保存最终的P、Q")
        P.tofile("train/P.bin")
        Q.tofile("train/Q.bin")
        self.all_score = np.dot(P,Q)
        self.all_score.tofile("train/all_score.bin")
        print("矩阵分解完成")
        # print(self.all_score)
        # self.get_recommand(self.user[0])

    def get_recommand(self, user_id):
        K = self.n_sim_movie  #相似的电影数目
        N = self.n_rec_movie    #要推荐的电影数目
        # print("user_id", user_id)
        #未评分电影：分数为0
        # not_score_movie = set(self.user_movie_maxtri[self.user.index(user_id)])
        not_score_movie = set()
        index_u = self.user.index(user_id)
        sort_movie = {}
        for i, score in enumerate(self.user_movie_maxtri[index_u]):
            if score==0:
                # not_score_movie.add(self.movie[i]) #获得该用户没有评分的电影
                sort_movie[self.movie[i]] = self.all_score[index_u][i]
        #排序这些没有评分的电影（依据评分高低进行排序）
        # for mov_id in not_score_movie:
            # sort_movie[mov_id] = all_score[index_u][self.movie.index(mov_id)]
        # print("sort_movie", sort_movie)
        # print(sorted(sort_movie.items(), key = itemgetter(1), reverse=True))
        return sorted(sort_movie.items(), key = itemgetter(1), reverse=True)[:K]
        
    def continue_train(self):#在中断训练之后，恢复训练
        # print("self.trainSet", self.trainSet)
        print("数据恢复中...")
        with open("trainSet31328.json", "r") as f:
            self.trainSet = json.load(f)
            f.close()
        with open("testSet10369.json", "r") as f:
            self.testSet = json.load(f)
            f.close()
        # self.user_movie_maxtri = np.zeros((len(self.u), len(self.m)), dtype=float)
        with open("user.txt", "r") as f:
            self.user = f.read().replace("'","").replace("(","").replace(")","").split(', ')
            f.close()
        with open("movie.txt", "r") as f:
            self.movie = f.read().replace("'","").replace("(","").replace(")","").split(', ')
            f.close()
        self.user_movie_maxtri = np.fromfile("user_movie_maxtri.bin", dtype=float)
        self.user_movie_maxtri.shape = len(self.user), len(self.movie)
        
        # self.user = tuple(self.u)
        # self.movie = tuple(self.m)
        print("user  len ",len(self.user))
        print("movie  len ",len(self.movie))
        loss_list = [] #存储每次迭代计算的loss值
        N = len(self.user)
        M = len(self.movie)
        K = N
        alpha=0.006
        beta=0.02
        if os.path.exists("P.bin") and os.path.exists("Q.bin"):
            P = np.fromfile("P.bin", dtype=float)
            Q = np.fromfile("Q.bin", dtype=float)
            P.shape = N,K
            Q.shape = K,M
        else:
            P = np.random.rand(N, K)
            Q = np.random.rand(M, K)
            Q = Q.T #Q矩阵转置
        print("数据恢复完成")
        print("开始矩阵分解")
        for step in range(5000):
            #更新R^
            for i in range(N):
                for j in range(M):
                    if self.user_movie_maxtri[i][j] != 0:
                        #计算损失函数
                        error = self.user_movie_maxtri[i][j]
                        for k in range(K):
                            error -= P[i][k]*Q[k][j]
                        #优化P,Q矩阵的元素
                        for k in range(K):
                            P[i][k] = P[i][k] + alpha*(2*error*Q[k][j]-beta*P[i][k])
                            Q[k][j] = Q[k][j] + alpha*(2*error*P[i][k]-beta*Q[k][j])
            P.tofile("P.bin")
            Q.tofile("Q.bin")
            print("计算loss")
            loss = 0.0
            #计算每一次迭代后的loss大小，就是原来R矩阵里面每个非缺失值跟预测值的平方损失
            for i in range(N):
                for j in range(M):
                    if self.user_movie_maxtri[i][j] != 0:
                        #计算loss公式加号的左边
                        data = 0
                        for k in range(K):
                            data = data + P[i][k]*Q[k][j]
                        loss = loss + math.pow(self.user_movie_maxtri[i][j]-data,2)
                        #得到完整loss值
                        for k in range(K):
                            loss = loss + beta/2*(P[i][k]*P[i][k]+Q[k][j]*Q[k][j])
                        loss_list.append(loss)
            #输出loss值
            print("loss={:}".format(loss))
            if (step+1) % 1000 == 0:
                print("loss={:}".format(loss))
            #判断
            if loss < 0.001:
                print(loss)
                break
        # print(loss_list)
        self.all_score = np.dot(P,Q)
        self.all_score.tofile("all_score.bin")
        print("矩阵分解完成")
    # 产生推荐并通过准确率、召回率和覆盖率进行评估
    def evaluate(self):
        print('Evaluating start ...')
        N = self.n_rec_movie
        # 准确率和召回率
        hit = 0
        rec_count = 0
        test_count = 0
        # 覆盖率
        all_rec_movies = set()

        for i, user in enumerate(self.trainSet):
            test_moives = self.testSet.get(user, {})
            # rec_movies = self.recommend(user)
            rec_movies = self.get_recommand(user)
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


if __name__ == '__main__':
    # rating_file = '..\\..\\ml-latest-small\\ratings.csv'    #用户评分表
    rating_file = 'data/rand_set7.csv'
    # rating_file = "/home/aistudio/hzm/ml-latest-small2/test.csv"
    itemCF = ItemBasedCF()

    itemCF.get_dataset(rating_file)

    itemCF.recommand_maxtri()
    # # itemCF.calc_movie_sim()

    # itemCF.evaluate()


    # itemCF.continue_train()
    print("end")







