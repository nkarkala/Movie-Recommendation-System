__author__ = 'niki'
import numpy as np
import similarity
from collections import OrderedDict
import time
import operator
from itertools import islice
import math
import copy

class Prediction:
    #initialize data structures
    def __init__(self, train_data, test_data):
        self.train=train_data
        self.test,self.predict=OrderedDict(), OrderedDict()
        self.avg,self.iuf=np.zeros(200),np.zeros(1000)
        for row in test_data:
            if 0 == row[2]:
                self.predict.setdefault(row[0], OrderedDict()).update({row[1]: row[2]})
            else:
                self.test.setdefault(row[0], OrderedDict()).update({row[1]: row[2]})
    '''
    Basic vector similarity

    '''
    #Return cosine similarity between a user in test set u1 and user in train u2
    def cosine_similarity(self,u1,u2):
        a,b=[],[]
        for movie in self.test[u1]:
                if 0 != self.train[u2][movie-1]:
                    a.append(self.train[u2][movie-1])
                    b.append(self.test[u1][movie])
                # If movie not rated by train  user u2 then append his average rating
                else:
                    a.append(self.avg[u2])
                    b.append(self.test[u1][movie])
        return similarity.similarity.cosine_similarity(a,b)

    #Make prediction for all users in test on basic cosine similarity
    def make_prediction_vector_similarity(self,result_file):
        self.calculate_iuf()
        self.calculate_user_avg()
        f = open(result_file, 'w')
        for user in self.predict:
            for movie in self.predict[user]:
                rating=self.cosine_prediction_user(user,movie)
                f.write(str(int(user)) + " " + str(int(movie)) + " " + str(rating) + "\n")
                self.predict[user][movie]=rating
        f.close()
        return self.predict

    #Predict user rating for movie using basic cosine
    def cosine_prediction_user(self,user,movie):
        num, den = 0.0, 0.0
        for k in range(len(self.train)):
            if self.train[k][movie - 1] != 0:
                weight=self.cosine_similarity(user,k)
                num += (weight* self.train[k][movie - 1])
                den += weight
        if den == 0.0:
            rating=3.0
        else:
            rating = (num / den)
        if(np.isnan(rating)):
            rating=3.0
        #rating = int(round(rating))
        if rating < 1:
            rating = 1.0
        elif rating >5:
            rating=5.0
        return rating

    '''
    Pearson Correlation

    '''
    #calculate user average for all users in train
    def calculate_user_avg(self):
        for user,user_row in enumerate(self.train):
            self.avg[user]=sum(user_row)/sum(user_row!=0)

    #Return pearson similarity between a user in test and user in train
    def pearson_similarity(self,u1,u2,use_iuf):
        a,b=[],[]
        u1_avg=(sum(self.test[u1].values()) / (len(self.test[u1])*1.0))
        u2_avg=self.avg[u2]
        for movie in self.test[u1]:
            if 0!=self.train[u2][movie-1]:
                if(use_iuf==False):
                      a.append((self.train[u2][movie-1]-u2_avg))
                      b.append((self.test[u1][movie]-u1_avg))
                else:
                      iuf=self.iuf[movie-1]
                      a.append((((self.train[u2][movie-1]*iuf-u2_avg))))
                      b.append((((self.test[u1][movie]*iuf-u1_avg))))
            else:
                 if(use_iuf==False):
                    a.append((self.avg[u2]-u2_avg))
                    b.append((self.test[u1][movie]-u1_avg))
                 else:
                   iuf=self.iuf[movie-1]
                   a.append((self.avg[u2]*iuf-u2_avg))
                   b.append((self.test[u1][movie]*iuf-u1_avg))
        return similarity.similarity.cosine_similarity(a,b)

    def pearson_prediction_user(self,user,movie,use_iuf):
        num,den=0.0,0.0
        for k in range(len(self.train)):
            if self.train[k][movie-1]!=0:
                weight=self.pearson_similarity(user,k,use_iuf)
                num+=(weight*(self.train[k][movie-1]-self.avg[k]))
                den+=abs(weight)
        avg_rating=(sum(self.test[user].values()) / (len(self.test[user])*1.0))
        if den==0.0:
            rating=avg_rating
        else:
            rating=avg_rating+ (num/den)
        if rating < 1:
            rating =1
        elif rating>5:
            rating=5
        return rating

    #Prediction using pearson
    def make_prediction_pearson_similarity(self,result_file):
        #self.test=self.increase_test()
        self.calculate_user_avg()
        f = open(result_file, 'w')
        for user in self.predict:
            for movie in self.predict[user]:
                rating=self.pearson_prediction_user(user,movie,False)
                f.write(str(int(user)) + " " + str(int(movie)) + " " + str(rating) + "\n")
                self.predict[user][movie]=rating
        f.close()
        return self.predict

    '''
    Inverse User Frequency

    '''
    #calculate inverse user frequency for all movies.
    def calculate_iuf(self):
        no_of_users=len(self.train)
        train=np.transpose(self.train)
        for movie,movie_row in enumerate(train):
            self.iuf[movie]=math.log(no_of_users/sum(movie_row!=0))

    def make_prediction_iuf_pearson_similarity(self,result_file):
        self.calculate_user_avg()
        self.calculate_iuf()
        f = open(result_file, 'w')
        for user in self.predict:
            for movie in self.predict[user]:
                # pass argument to pearson_prediction as true to indicate to use IUF
                rating=self.pearson_prediction_user(user,movie,True)
                f.write(str(int(user)) + " " + str(int(movie)) + " " + str(rating) + "\n")
                self.predict[user][movie]=rating
        f.close()
        return self.predict

    '''

    Case Amplification for Pearson Correlation.

    '''
    def make_prediction_case_amp_pearson_similarity(self,result_file):
        self.calculate_user_avg()
        f = open(result_file, 'w')
        for user in self.predict:
            for movie in self.predict[user]:
                # pass argument to pearson_prediction as true to indicate to use IUF
                rating=self.pearson_case_amp_prediction_user(user,movie,False)
                f.write(str(int(user)) + " " + str(int(movie)) + " " + str(rating) + "\n")
                self.predict[user][movie]=rating
        f.close()
        return self.predict

    def pearson_case_amp_prediction_user(self,user,movie,use_iuf):
        num,den=0.0,0.0
        for k in range(len(self.train)):
            if self.train[k][movie-1]!=0:
                weight=self.pearson_similarity(user,k,use_iuf)
                weight=weight*(abs(weight)**1.5)
                num+=(weight*(self.train[k][movie-1]-self.avg[k]))
                den+=abs(weight)
        avg_rating=(sum(self.test[user].values()) / (len(self.test[user])*1.0))
        if den==0.0:
            rating=avg_rating
        else:
            rating=avg_rating+ (num/den)
        if rating < 1:
            rating = 1.0
        elif rating>5:
            rating=5
        return rating

    '''
    Item Based Approach

    '''
    def pearson_similarity_items(self,m1,m2):
        movie_row1=self.train[:,m1-1]
        movie_row2=self.train[:,m2-1]
        '''
        To Subtract movie rating instead of User
        avg_m1=sum(self.train[:,m1-1])/sum(self.train[:,m1-1]!=0)
        avg_m2=sum(self.train[:,m2-1])/sum(self.train[:,m2-1]!=0)
        if(np.isnan(avg_m1)):
            avg_m1=3.0
        if(np.isnan(avg_m2)):
            avg_m2=3.0
        '''
        a,b=[],[]
        for i in range(0,200):
            a.append((movie_row1[i])-self.avg[i])
            b.append((movie_row2[i])-self.avg[i])
        return similarity.similarity.cosine_similarity(a,b)

    def make_item_based_prediction(self,result_file):
        self.calculate_user_avg()
        # Make prediction block by block
        f = open(result_file, 'w')
        for user in self.predict:
            user_avg=sum(self.test[user].values()) / (len(self.test[user])*1.0)
            for movie in self.predict[user]:
                #Fetch Movies that user has already rated(test data)
                num,den=0.0,0.0
                for m in self.test[user]:
                    #Similarity between the movies
                    weight=self.pearson_similarity_items(movie,m)
                    num+=(self.test[user][m]*weight)
                    den+=abs(weight)
                if(den==0.0):
                    rating=3.0
                else:
                    rating=num/den
                rating=int(round(rating))
                if(rating<1):
                    rating=1
                elif(rating>5):
                    rating=5
                f.write(str(int(user)) + " " + str(int(movie)) + " " + str(int(round(rating))) + "\n")
                self.predict[user][movie]=rating
        return self.predict

    # Increase Test Data by providing predictions for most rated movies.
    def increase_test(self):
        test2=copy.deepcopy(self.test)
        self.calculate_user_avg()
        self.calculate_iuf()
        index=201
        num_ratings={}
        for movie in range(1000):
            count=sum(self.train[:,movie]!=0)
            num_ratings[movie]=count
        num_ratings = OrderedDict(sorted(num_ratings.items(), key=operator.itemgetter(1),reverse=True))
        for user in range(0,100):
            c=0
            for movie in islice(num_ratings.keys(),5):
                if(self.predict.get(index+user)):
                   if(self.test[index+user].get(movie+1)==None and self.predict[index+user].get(movie+1)==None):
                     p=self.pearson_prediction_user(index+user,movie+1,False)
                   p=self.pearson_prediction_user(index+user,movie+1,True)
                   test2.setdefault(index+user, OrderedDict()).update({movie+1:p})
        return(test2)


#Custom Method to combine 3 methods and take weighted average.

start_time = time.time()
train = np.array(np.loadtxt("train.txt"))
test = np.loadtxt("test5.txt")
p=Prediction(train,test)
predict1=copy.deepcopy(p.make_prediction_vector_similarity("results/vector.txt"))
predict2=copy.deepcopy(p.make_prediction_pearson_similarity("results/pearson.txt"))
predict3=copy.deepcopy(p.make_prediction_iuf_pearson_similarity("results/iuf.txt"))

f = open("results5.txt", 'w')
for user in predict2:
    for movie in predict2[user]:
        rating=int((round((0.5*predict1[user][movie])+(0.3*predict2[user][movie])+(0.2*predict3[user][movie]))))
        if(rating<1):
            rating=1
        elif(rating>5):
            rating=5
        predict2[user][movie]=rating
        f.write(str(int(user)) + " " + str(int(movie)) + " " + str(rating) + "\n")
f.close()
print("Time taken : %s seconds:" %(time.time() - start_time))







