import os,inspect
import warnings
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"]='0'
warnings.filterwarnings('ignore')

import tensorflow as tf
import helper.data_builder as db
import helper.dataloder as dl
import helper.model_builder as mb
import helper.train as trainner


PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
CKPT_DIR = PACK_PATH+'/Checkpoint'


def main():
    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass

    db.DataBuilder()
    dataset = dl.Dataset()
    model_object = mb.Model()
    
    model = model_object.compelete_model(FLAGS.model_size)
    print(f"model summary \n{model.summary()}")

    model_object.visualize_model()
    print("Visualizing model PNG saved")

    optim = tf.keras.optimizers.Adam(learning_rate=FLAGS.lr)
    model.compile(loss="categorical_crossentropy",optimizer=optim,metrics=["accuracy"])

    trainner.train(model,FLAGS.epoch,FLAGS.batch_size,dataset,chkpt_dir = CKPT_DIR)




if __name__ == "main":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch",type=int,default=10,help="Training Epochs")
    parser.add_argument("--lr",type=float,default=1e-3,help="Initial Learning Rate")
    parser.add_argument("--batch_size",type=int,default=32,help="Training Batch Size")
    parser.add_argument("--model_size",type=int,default=32,help="width of first layer of resnet block subsequent layers are +32 than this")
    FLAGS, unparsed = parser.parse_known_args()

    main()