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
 