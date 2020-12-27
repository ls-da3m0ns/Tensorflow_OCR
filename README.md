# Tensorflow OCR
Perform OCR on alphabets and digits using tensorflow 

## Usage
``` python run.py --epoch 20 --lr 1e-4 --batch_size 64 --model_size 32 ```
<br>
## Requirements
 * Tensorflow 2.2.0
 * Numpy      3.7.6
 * Opencv     
 * gdown
 * sklearn
 * Tensorboard

## DataSet
I have used combination two datasets 
 * Mnist Digits (contains 0-9 digits)
 * NIST special Database 19  (contains a-z alphabets)

## Model 
We have Used ResNet styled architechture for better perfomance 

<div align="center">
<p>Model Architecture.</p>
  <img src="./model.png" width="800">  
</div>