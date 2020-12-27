import os
import numpy as np 
import gdown
import tensorflow as tf

class DataBuilder():
    def __init__(self):
        print("DataBuilder called \nStarting Gathering Data ...")
        #os.system("wget https://www.transfernow.net/Q1QvxL122020")
        if os.path.exists("./A_Z Handwritten Data.csv") is False:
            gdown.download("https://drive.google.com/u/0/uc?id=1YWcKWGrLdq1QD9yM7OeReG6ZEM-dNjkK&export=download","archive.zip",quiet=True)
            os.system("unzip archive.zip")

        if os.path.exists("./temp_files") is False:
            os.mkdir("./temp_files")
        
        
        print("combine and save process started")
        self.combine_dataset()
        print("combine and save process compelete")
    
    def a2z_data_setup(self,dataset_path="./A_Z Handwritten Data.csv"):
        data = []
        labels = []
        for row in open(dataset_path):
            row = row.split(",")
            label = int(row[0])
            image = np.array([int(x) for x in row[1:]], dtype="uint8")
            image = image.reshape((28,28))
            data.append(image)
            labels.append(label)

        data = np.array(data,dtype="float32")
        labels = np.array(labels,dtype="int")
        return data,labels
    
    def zero2nine_data_setup(self):
        ((train_images,train_labels),(test_images,test_labels)) = tf.keras.datasets.mnist.load_data()
        data = np.vstack([train_images,test_images])
        labels = np.hstack([train_labels,test_labels])
        return data,labels
    
    def combine_dataset(self):
        if os.path.exists("./temp_files/data.npy") and os.path.exists("./temp_files/labels.npy"):
            return 
        data_a2z,label_a2z = self.a2z_data_setup()
        data_ze2nn,label_ze2nn = self.zero2nine_data_setup()

        label_a2z += 10

        data = np.vstack([data_a2z,data_ze2nn])
        labels = np.hstack([label_a2z,label_ze2nn])
        
        print("saving data and labels as data.npy and labels.npy in temp_files/")
        np.save("./temp_files/data.npy",data)
        np.save("./temp_files/labels.npy",labels)