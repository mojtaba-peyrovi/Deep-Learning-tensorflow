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

see the code in file: RNN_implementation.ipynb

#### Word2Vec library:
This library, is a useful library for NLP analysis and it vectorizes words into vectors. 

Words are another type of recurrent neural network uses.

In classic NLP words are typically replaced by numbers that show the frequency of the word in the sentence. Doing this approach we will lose information about the words relationship among themselves.

##### Two approached of NLP:

1) __Count-Based:__ frequency of words in corpus (متن)
2) __Predictive-Based:__ Neighboring words are predicted based on a vector space.


`Word2Vec ueses the predictive approach.`

In this tutorial we learn one of the most important uses of Neural Networks, which is __Word2Vec model created by Thomas Mikolov et al.__

The goal is to learn word embeddings by modeling each word as a vector in n-dimensional space.

__EMBEDDING:__ An embedding is a mapping of a discrete — categorical — variable to a vector of continuous numbers. In the context of neural networks, embeddings are low-dimensional, learned continuous vector representations of discrete variables. Neural network embeddings are useful because they can reduce the dimensionality of categorical variables and meaningfully represent categories in the transformed space.

##### Why do we use word-embedding?

The reason, is because the vector representation of words in a n-dimensional space will be so sparce and we can't use them in the way we could use vectors for audio analysis, or image analysis.

Word2Vec creates vectors spaced models that represent (embed) words in a continuous vector space.

Using this approach, we can perform vector mathematics on words (add/subtract, or check similarity.)

- During the training, similar words will find their vectors closer together.
- The model may produce axes that represent concepts, such as gender, verbs, singular vs plural, etc.
(word2vec-word-relationship-examples.jpg)

Word2Vec comes in two different algorithms:

1) __Skip-Gram model:__ better for larger datasets. Finds the context around a specific word.
2) __CBOW(Continuous Bag Of Words):__ better for smaller datasets. Finds the word for a specific context.

#####Let's start with CBOW: 

The training mechanism is called __Noise Contrastive Training__ and the goal of it, is to return high probability for the correct word, and the low probability for other words.

Although using TF is powerful or making models of Word2Vec, but it can be so hard. so there is another library built on the top of TF, called [Gensim](https://radimrehurek.com/gensim/models/word2vec.html) that has a much simpler API.



