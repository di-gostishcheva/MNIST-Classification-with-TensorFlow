# MNIST-Classification-with-TensorFlow
This project is developed to train and test a neural network on TensorFlow for classifying handwritten digits from the MNIST dataset. The neural network consists of 3 Dense layers: two hidden layers, each with 128 neurons, and an output layer. The activation functions on the hidden layers are relu, and softmax on the output layer.


## Description

This project is developed to train and test a neural network on TensorFlow for classifying handwritten digits from the MNIST dataset. The neural network consists of 3 Dense layers: two hidden layers, each with 128 neurons, and an output layer. The activation functions on the hidden layers are relu, and softmax on the output layer.

## Used Libraries

- TensorFlow
- Matplotlib
- NumPy
- Scikit-learn
- Keras

## Data Loading

For this project, the MNIST dataset is used, which is loaded from Keras.datasets. The dataset consists of 60,000 training images and 10,000 test images of handwritten digits, each of size 28x28 pixels.

## Data Preparation

Each pixel of the image is represented by an integer from 0 to 255. Before training the model, all pixel values are normalized by dividing by 255.

The target variable is also converted to a one-hot format, where the true value of the target variable is represented by a vector of length 10, consisting of zeros with a single one in the position corresponding to the class to which the image belongs.

## Model Creation

The neural network model is created using the Keras Sequential API. It includes a Flatten layer to transform the two-dimensional image into a one-dimensional vector, two hidden Dense layers with 128 neurons each, and an output Dense layer with 10 neurons for predicting the probabilities of each class.

## Model Training

The neural network is trained on the training dataset using the categorical_crossentropy loss function and the adam optimizer. Training is conducted for 5 epochs with a batch size of 40.

## Model Evaluation

After training, the model is evaluated on the test dataset to assess its performance. Accuracy and loss metrics are displayed.

## Model Architecture

```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 flatten (Flatten)           (None, 784)               0         
                                                                 
 dense (Dense)               (None, 128)               100480    
                                                                 
 activation (Activation)     (None, 128)               0         
                                                                 
 dense_1 (Dense)             (None, 128)               16512     
                                                                 
 activation_1 (Activation)   (None, 128)               0         
                                                                 
 dense_2 (Dense)             (None, 10)                1290      
                                                                 
 activation_2 (Activation)   (None, 10)                0         
                                                                 
=================================================================
Total params: 118,282
Trainable params: 118,282
Non-trainable params: 0
_________________________________________________________________
```

Answer: Trainable params: 118, 282

## Model Parameters

- Loss function: categorical_crossentropy
- Optimizer: Adam
- Metric: accuracy

## Environment Requirements

- TensorFlow 2.x
- Matplotlib
- NumPy
- Scikit-learn
- Keras

