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

