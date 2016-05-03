__author__ = 'niki'
import numpy as np

class similarity:
    @staticmethod
    def cosine_similarity(a, b):
        a,b=np.array(a),np.array(b)
        if a.size==0:
            return 0.0
        sim=sum(a*b)/((sum(a**2)*sum(b**2))**(1/2.0))
        if(np.isnan(sim)):
            sim=0.0
        return sim

    @staticmethod
    def pearson_similarity(a, b):
        a, b = np.array(a), np.array(b)
        a = a - (a.sum() / len(a))
        b = b - (b.sum()) / len(b)
        if len(a)<1:
           sim=0.0
        else:
            sim=sum(a*b)/((sum(a**2)*(sum(b**2)))**(1/2.0))
        if(np.isnan(sim)):
            sim=0.0
        return sim

    @staticmethod
    def euclidean_distance(a,b):
        a,b=np.array(a),np.array(b)
        sim=(sum((a-b)**2))**(1/2.0)
        den=(16*len(a))**(1/2.0)
        return 1-sim/den



