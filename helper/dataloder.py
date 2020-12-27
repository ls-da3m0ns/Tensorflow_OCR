import numpy as np, os, cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split

class Dataset():
    def __init__(self):
        print("Dataset Called ...")
        self.data = None
        self.labels = None
        self.trainX,self.trainY = None,None
        self.testX,self.testY = None,None

        self.train_btindex = 0
        self.test_btindex = 0

        print("loading data ...")
        self.load_data()

        print("preprocessing data ...")
        self.preprocess_data(normalise = True)


        print("generating class weights data ...")
        self.classWeights = self.get_classWeights()

        print("spliting data ...")
        self.split_data()

        self.num_train = self.trainX.shape[0]
        self.num_test = self.testX.shape[0]

        del self.data
        del self.labels


    def preprocess_data(self,normalise=True):
        self.data = np.array([cv2.resize(image,(32,32)) for image in self.data],dtype="float32")
        self.labels = np.array([tf.one_hot(lbl,depth=36) for lbl in self.labels],dtype="int")
        
        if normalise:
            self.data /= 255.0

        self.data = np.expand_dims(self.data,axis=-1)


    def load_data(self):
        if os.path.exists("./temp_files/data.npy") and os.path.exists("./temp_files/labels.npy"):
            self.data = np.load("./temp_files/data.npy")
            self.labels = np.load("./temp_files/labels.npy")
        else:
            raise FileNotFoundError

    def split_data(self):
        self.trainX,self.testX,self.trainY,self.testY = train_test_split(self.data,self.labels,test_size=0.20,stratify=self.labels,random_state=42)
    
    def get_classWeights(self):
        count = self.labels.sum(axis=0)
        classWeight = {}
        for i in range(0,len(count)):
            classWeight[i] = count.max()/count[i]
        return classWeight

    def next_train_batch(self,batch_size=1,terminator=False):
        start = self.train_btindex
        if (self.train_btindex+batch_size >= self.num_train):
            end = self.num_train
            terminator = True
        else:
            end = self.train_btindex + batch_size
            self.train_btindex += batch_size
        
        return self.trainX[start:end],self.trainY[start:end],terminator
    
    def next_test_batch(self,batch_size=1,terminator=False):
        start = self.test_btindex
        if (self.test_btindex+batch_size >= self.num_test):
            end = self.num_test
            terminator = True
        else:
            end = self.test_btindex + batch_size
            self.test_btindex += batch_size
        
        return self.testX[start:end],self.testY[start:end],terminator
         
    def get_full_test(self):
        return self.testX, self.testY 