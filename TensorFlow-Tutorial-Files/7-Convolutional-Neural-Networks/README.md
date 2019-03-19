### Intro to Convolutional Neural Network  (CNN)

---
####Review:
 There are several types of activation functions for Neural Networks. 
 
 1) __Perceptrons:__ Checks if the input is positive or negative.
 2) __Sigmoid__
 3) Hyperbolic Tangent (Tanh)
 4) Rectified Linear Unit (reLu)
  
 We will see more later on.
 
 __Cost Function types:__
 
 1) Quadratic Cost Function
 
 2) Cross Entropy
 
 #### Weight initialization:
 
 There are options for initialization of weights, for example putting all zero. but it doesn't have the randomization benefit. or other items are not so proficient. A good approach is:
 
 ##### Xavier(Glorot) initialization:
 photo: xavier-initialization.jpg
 
 W: is the variance of the distribution of weights initialized first. 
 n(in): is the number of neural feeding in the network.
 
 We don't need to learn the math behind it, just use it with TensorFlow.
 
 ####Review Some Terms:
 
 1) __Learning Rate__: The size of steps during gradient descent. If we pick a very small number, it may take a very long time to converge to the minimum, or never converge. Also, if we pick a large number, we make end up overshooting towards the minimum and never converge.
 
 2) __Batch Size:__ Batched allow us to split data into batches and then using stochastic gradient descent. 
 
 3) __Second Order Behavior of Gradient Descent:__
 
 It is a method that makes the conversion so much faster, but having the steps smaller as we get closer to the minimum. There are some algorithms doing this such as :
    
     AdaGrad
     RMSProp
     Adam
 
 Adam is an important one we are going to use here.
 
 4) __Overfitting and underfitting:__ Underfitted models, have high errors on both test/train data whereas overfitting models, perform so well on training data, but very bad on test data.
 
 With potentially hundreds of parameters in a deep neural network, the possibility of overfitting is very high. We will see ways to control it. 
 
 - One of the ways to handle this in Neural Networks theory, is called __Dropout.__  In this method, we will randomly throw neurons away from the model, so that we know we don't rely on a specific neuron.
 - Another method is called __Expanding data__.  we artificially expand data by adding noise, tilting image, adding low white noise to the sound, etc.
 
#### MINST dataset:

It has 55,000 training images, 10,000 test images, and 5000 validation images.

It has handwriting samples of 0-9 by their pixels value. (0 for white, 1 for black, and other values between 0-1 for other shades of gray.)

Each digit is represented as a 2d array, size 28x28. (photo:each-mnist-digit.jpg)

First we need to flatten the arrays to a 1-d array size (784,1) or (1,781) which is 28*28.

Now we will have all the training images, as a single tensor of 55000 vectors, size of 781 each. (photo: single-training-tensor.jpg)

__One-Hot-Encoding:__ It is a method to show the final results. Instead of having the results as labels 'one', 'two', etc. we just make a matrix of [0 0 0 1 0 0 0 0 0 0] which in this case means the number is 3. 

Based on one-hot-encoding method, the labels(results) for the dataset, will be a tensor of 10x55000 size. (photo: final-tensor.jpg)


__Softmax Regression:__ It returns a list of values btween 0 and 1, that adds up to 1, so we can use them as a list of probabilities. For example, if we have ten labels to predict, it returns a series of ten probability that shows how much is each label's probability and which one has the highest chance.

In CNN, we can use softmax regression as our activation function. (photo:softmax-function.jpg, and softmax-calculation,jpg)

##### BASIC APPROACH:

Now we can see the basic approach to solve the MNIST problem and return the predictions. 

First we import data, and we can visualize it using matplotlib. 

__IMPORTANT:__ Before plotting a sample image, we need to reshape the sample, because each sample in the dataset, is (1,784) dimension, we need to make it (28,28). Because the dataset samples are saved as Numpy arrayx, we can use reshape() method to convert them to 20x20. 

- When we want to convert labels like true/false labels to 0/1, we can uae this:
```
tf.cast(<data_set_with_strings>, <numerical_Datatpye>)\
example: tf.cast(correct_prediction, float32)  // converts true/falses into 0/1
```



  
