### Recurrnet Neural Networks(RNN):
---
Redcurrent neural networks are used for dealing with sequential data such as time series.

Some examples of subsequent data:

- Stock market data
- sentences
- audio
- car trajectories (مسیر های اتوموبیلها روی نقشه)
- music

we can imagine a vector that each index number shows a specific timing of each index, and the value of the index shows the value that happened in that specific time. like [1,2,3,4,5,6]

The whole theory is, to predict the next number after 6. In simple examples, we can easily say 7 because the sequence will be one by one in this example. so we can predict the next 6 numbers are [2,3,4,5,6,7]

But when it gets complicated, we would love to have model that predicts the next number.

#### The difference between a regular ANN, and RNN:

For a typical neuron, we have the input, activation function, and output. (photo: typical_neuron.jpg)

But when we have a RNN, we will have the output feeding back the same neuron. (photo:recurrent-neuron.jpg)

Here is a good visualization that shows how a single neuron gets fed during a period of 3 units. (recurrent-neuron-during-period.jpg)

The photo above shows that each neuron takes two sets of inputs, one from its own time, one from the previous time's neuron.

The neurons that are the function of inputs from the previous time steps, are known as __Memory Cells.__

#### Types of RNN in terms of input/output types:

1) Sequence to Sequence: The input and output are both sequential. For example, the input is the sales for each day in a month, and trying to predict the sales for the whole days of the next month. 

(sequence-to-sequence-rnn.jpg)

2) Sequence to Vector: This happens when we want to accumulate the results of the previous time series and make a conclusion. A good example of it, is sentiment analysis. (sequence-to-vector-rnn.jpg )

3) Vector to Sequence: When we have only one input, and the results will be a sequence of values. for example, if we input a single photo, and the system returns a series of different captions for that photo . (vector-to-sequence-rnn.jpg)


#### The manual example:

Here we do a simple recurrent example, that has a layer of 3 neurons. check the photos: (manual-example-1,2,3)


There are some challenges happen while working with recurrent neural networks. 
 
 
 __Vanishing Gradient:__ It happens when on the backpropogation process, the gradient vanishes or explodes. It happnes mostly with big networks. when backpropagating, the gradient gets smaller and smaller, it causes the weights never change at lower levels. The opposite is possible too. The gradient gets bigger and bigger, and explodes.

There are some solutions on it:

- first, using different activation functions that can help, liks ReLu, Leaky ReLu, ELU.
- Another solution, is to do batch normalization that the model normalizes each batch using batch mean and standard deviation.
- Gradient clipping: gradients are cut off before a predetermined limits.

 
#### Special neurons for time-series analysis:

we discuss two types of neurons. __Long Short-Term Memory (LSTM)__ and __Gated Recurrent Units (GRU).__ 

The technical stuff behind the scene has been covered in TensorFlow and we don't have to worry about them. The nice and clean API of TF will be used and explained here.

#### Predicting the sequential values of the function sin(x):

photo: sin-x-prediction.jpg

Now we are going to train the model, using a sequence, and try to predict one step further along the sequence. 

Later we try to predict way more steps further along the sequence based on seed series( in this example all zeros)



