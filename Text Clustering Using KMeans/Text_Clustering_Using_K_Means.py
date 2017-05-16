
# coding: utf-8

# In[17]:

import pandas as pd
import scipy
import numpy as np
from io import StringIO
from collections import Counter
from scipy.sparse import csr_matrix
import random


# In[18]:

#Read the train data and save it in dataframe after splitting by new lines
with open('train.dat','r') as f:
    df = pd.DataFrame(l.split("\t") for l in f) 
df.columns = ["term-freq"] 


# In[19]:

from sklearn.metrics.pairwise import cosine_similarity

def getLebelsAndFreq(r):       
        c = len(r)
        lebel=[]
        frequency =[]
        for i in range(c):
            if(i%2 ==0):
                lebel.append(r[i])
                if(i<c):
                    frequency.append(r[i+1])
        return lebel,frequency  

def get_mode(l):
    sum1 = sum( i*i for i in l)
    return (sum1**0.5)

def get_dotProduct(l1,f1,l2,f2,common_lebel):
    dot_product = 0
    for i in common_lebel:
        loc1 = l1.index(i)
        floc1 = f1[loc1]
    
        loc2 = l2.index(i)
        floc2 = f2[loc2]
        dot_product = dot_product + (floc1 * floc2)
    return dot_product
    
def getCosineSimilarity(d1,d2):
    freq1 = d1.split()  
    r1 = [float(i) for i in freq1]    
        
    freq2 = d2.split()  
    r2 = [float(i) for i in freq2]

    
    l2,f2 = getLebelsAndFreq(r2)
    l1,f1 = getLebelsAndFreq(r1)

    common_lebel =set(l2) & set(l1)
    common_lebel = list(common_lebel)
    
    dot_product = get_dotProduct(l1,f1,l2,f2,common_lebel)
    mode = get_mode(l1) * get_mode(l2)
    cos_sim = (dot_product/mode)
    return cos_sim

def getEuclideanDistance(d1,d2):
    freq1 = d1.split()  
    r1 = [float(i) for i in freq1]    
        
    freq2 = d2.split()  
    r2 = [float(i) for i in freq2]

    
    l2,f2 = getLebelsAndFreq(r2)
    l1,f1 = getLebelsAndFreq(r1)

    common_lebel =set(l2) & set(l1)
    common_lebel = list(common_lebel)
    
    summation = 0
    for i in common_lebel:
        loc1 = l1.index(i)
        floc1 = f1[loc1]
    
        loc2 = l2.index(i)
        floc2 = f2[loc2]
        summation =  (floc1-floc2)**2
    sqrt_sum = (summation ** 0.5)    
    return sqrt_sum
    
    

def getAverage(clf,doc_count):
    feature_dict={}
    if(clf):
        d = clf[0].split(' ')
        l = len(d)
        for i in range(l):
            old_value = 0.0
            v = 0.0
            if(i%2==0 and i<l):
                k = d[i]
                v = d[i+1]
                if(k in feature_dict):
                    old_value = feature_dict.get(k)
                    feature_dict[k] = float(old_value) + float(v)                     
                else:
                    feature_dict[k] = float(v)
    avg_list=[]
    for k,v in feature_dict.items():
        avg_list.append(k)
        avg_freq = float((v/doc_count))
        avg_list.append(avg_freq)
           
    if(len(avg_list)>0):
        strg =''
        for i in avg_list:
            strg = strg +' '+str(i)
        strg = strg.strip()    
        TESTDATA=StringIO(strg)
        df1 = pd.read_csv(TESTDATA)
        return df1.columns.values
        
def getDifference(cc,pc):
    if (cc is None or pc is None):
        return 0
    elif(len(cc)==0 or len(pc)==0):
        return 0
    else:
        listpc = pc.split(' ')
        c = len(listpc)
        c= c-1
        lebel_pc=[]
        for i in range(c):
            if(i%2 ==0):
                lebel_pc.append(listpc[i])
                
        listcc = cc[0].split(' ')
        d = len(listcc)
        d= d-1
        lebel_cc=[]
        for i in range(d):
            if(i%2 ==0 and i<d):
                lebel_cc.append(listcc[i+1])
        common_lebel = set(lebel_pc).intersection(lebel_cc)
        return (len(common_lebel))
             


# In[20]:

def getCosineSimilaritySecondTime(d1,d2):
    freq1 = d1.split()  
    r1 = [float(i) for i in freq1]    
    l1 = len(d2)
    if(l1==1):
        all_one = d2[0]
        l_one = all_one.split(' ')
        r2 = [float(i) for i in l_one]   
    else:
        freq2 = d2.split(' ')  
        r2 = [float(i) for i in freq2]
    l2,f2 = getLebelsAndFreq(r2)
    l1,f1 = getLebelsAndFreq(r1)

    common_lebel =set(l2) & set(l1)
    common_lebel = list(common_lebel)
    
    dot_product = get_dotProduct(l1,f1,l2,f2,common_lebel)
    mode = get_mode(l1) * get_mode(l2)
    cos_sim = (dot_product/mode)
    return cos_sim

def getEuclideanDistanceSecondTime(d1,d2):
    freq1 = d1.split()  
    r1 = [float(i) for i in freq1]    
    l1 = len(d2)
    if(l1==1):
        all_one = d2[0]
        l_one = all_one.split(' ')
        r2 = [float(i) for i in l_one]   
    else:
        freq2 = d2.split(' ')  
        r2 = [float(i) for i in freq2]
    l2,f2 = getLebelsAndFreq(r2)
    l1,f1 = getLebelsAndFreq(r1)

    common_lebel =set(l2) & set(l1)
    common_lebel = list(common_lebel)
    
    summation = 0
    for i in common_lebel:
        loc1 = l1.index(i)
        floc1 = f1[loc1]
    
        loc2 = l2.index(i)
        floc2 = f2[loc2]
        summation =  (floc1-floc2)**2
    sqrt_sum = (summation ** 0.5)    
    return sqrt_sum

def getDifferenceSecondTime(cc,pc):
    if (cc is None or pc is None):
        return 0
    elif(len(cc)==0 or len(pc)==0):
        return 0
    else:
        listpc = pc[0].split(' ')
        c = len(listpc)
        c= c-1
        lebel_pc=[]
        for i in range(c):
            if(i%2 ==0 and i<c):
                lebel_pc.append(listpc[i])
                
        listcc = cc[0].split(' ')
        d = len(listcc)
        d= d-1
        lebel_cc=[]
        for i in range(d):
            if(i%2 ==0 and i<d):
                lebel_cc.append(listcc[i])
        
        common_lebel = set(lebel_pc).intersection(lebel_cc)
        return (len(common_lebel))


# In[21]:

#Kmeans using centroid distance 
class K_Means:
    def _init_(self, k =7, tol =500, max_itr = 30):
        self.k = k
        self.tol = tol
        self.max_itr = max_itr
        
    def fit(self,data,k,iteration):
        output = open("output.dat", 'w') 
        self.centroids = {}
        
        for i in range(k):
            d = data["term-freq"][i]
            self.centroids[i] = d
            optimized_set = set()
        for i in range(iteration):
            print("Iteration number is "+str(i))
            #print('Already Optimized Centroids')
            #print(optimized_set)
            self.classifications = {}
            cluster_count ={}
            output = open("output.dat", 'w') 
            for j in range(k):
                self.classifications[j] = [] 
                distances = []
            for featureset in data["term-freq"]:
                if(i==0):
                    distances = [getCosineSimilarity(featureset,centroid) for centroid in list(self.centroids.values())]
                else:
                    distances = [getCosineSimilaritySecondTime(featureset,centroid) for centroid in list(self.centroids.values())]
                classification = distances.index(min(distances)) 
                clt = classification + 1
                output.write(str(clt)+'\n')
                self.classifications[classification].append(featureset)
                if(classification in cluster_count):
                    old_value = cluster_count.get(classification)
                    cluster_count[classification] = float(old_value) + 1.00                     
                else:
                    cluster_count[classification] = 1
                
            prev_centroids = dict(self.centroids)

            for clf in self.classifications:
                clff = getAverage(self.classifications[clf],cluster_count.get(clf))
                if(clff is None):
                    pass
                else:
                    self.centroids[clf] = clff
            optimized = True    
            output.close()
            
            for c in self.centroids:
                previous_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if(i==0 ):
                    if getDifference(current_centroid,previous_centroid)< 30:
                        #print('Not Optimized')
                        if(i==2):
                            open('output.dat', 'w').close()
                        optimized = False
                    else:
                        optimized_set.add(c)
                else:
                    if getDifferenceSecondTime(current_centroid,previous_centroid)< 30:
                        #print('Not Optimized')
                        if(i==2):
                            open('output.dat', 'w').close()
                        optimized = False
                    else:
                        optimized_set.add(c)
    
            if optimized:
                break; 
        output.close()  


# In[22]:

clf = K_Means()
clf.fit(df,7,1)


# In[ ]:



