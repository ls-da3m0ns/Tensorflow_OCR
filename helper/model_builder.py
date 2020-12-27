import tensorflow as tf
from tensorflow import keras as keras
from tensorflow.keras.layers import *

class Model():
    def __init__(self) -> None:
        print("\nConstructing Model ... ")

    def compelete_model(self,model_size):
        input = Input(shape=(32,32,1))
        x = Conv2D(model_size,7,strides=1,padding="valid")(input)

        x = self.resnet_block(x,model_size,5,1)
        x = self.resnet_block(x,2*model_size,5,1)
        x = MaxPool2D()(x)

        x = self.resnet_block(x,4*model_size,3,2)
        x = self.resnet_block(x,8*model_size,3,1)
        x = MaxPool2D()(x)

        x = GlobalAveragePooling2D()(x)
        out = self.head(x)

        self.model = keras.Model(inputs=input,outputs=out)
        
        return self.model
    
    def resnet_block(self,input,f=32,k=5,s=2):
        x = Conv2D(f,kernel_size=k,strides=(s))(input)

        if k<3:
            k=3
        
        x = Conv2D(f,kernel_size=k-2,strides=(1),padding="same")(x)
        x = BatchNormalization()(x)

        y = self.skip_block(input,f*2,k,s)
        x = Concatenate()([x,y])

        return x

    def skip_block(self,input,f=32,k=3,s=2):
        x = Conv2D(f,kernel_size=k,strides=(s))(input)
        return x

    def head(self,input):
        x = Dense(128,activation="relu")(input)
        x = Dense(36,activation="sigmoid")(x)
        return x
    
    def visualize_model(self):
        tf.keras.utils.plot_model(
            model, to_file='model.png', show_shapes=True, show_dtype=True,
            show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96    )
