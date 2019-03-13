# TENSORFLOW
(Udemy course: Complete guide to TensorFlow for deep learning with python)
### TensorFlow basic Operations

---

__Tensor:__ is the fancy word for n-dimensional array.

for making a session:
```
with tf.Session() as sess:
    result = sess.run(a+b)   //this code adds a and b together and saves them in result variable
```

__Constant:__ constant variable. like numbers or arrays.

In order to make a matrix with a specific value in each cell:
```
fill_mat = tf.fill(<dimension>, <value>)  // tf.fill((2,4),10)
```

In the same way, we have tf.zeros() and tf.ones() that fills the matrix with 0 or 1 values.

- in order to fill values with random values from normal distribution, we say:
```
rand = tf.random_normal((4,4), mean=1, stddev=0.2)
```

- There is another type of random values like this:
```
rand = tf.random_uniform((4,4), minval=2, maxval=35)
```

__USEFUL TIP:__ while using notebooks, we can use interactive session, instead of normal session, to run the session along multiple cells. It doesn't have any use in other environments or IDE's.

we just need to say:
` sess = tf.InteractiveSession()`

and we address sess anytime we like during the notebook.

Sometimes sess.run() works the same way as sess.eval().

In order to see the shape of a matrix:
```
matrix_name.get_shape()
```

### TensorFlow Graphs:

---

TF uses graphs so much. Graphs are a set of nodes connected together called __Vertices.__

The connections to the nodes are called __Edges.__

In TF each node is an operation with possible input that will supply some output.

We can define graphs like this:

```
g = tf.Graph()
```

we can also have a graph as default graph:
```
dg = tf.get_default_graph()
```

In order to make an existing graph as default and check if it has changed:
```
with graph_two.as_default():
    print(graph_two is tf.get_default_graph())   // returns True
```

### TensorFlow Variables and Placceholders:

These are two types of tensor objects in a graph.

__Variables:__ During the optimization process, tf tunes the parameters of the model.

variables can hold the values of weights and biases during the session.

variables need to be initialized.

__Placeholders:__ Placeholders are initialized empty and used to feed in the actual training examples.

Placeholders need a declared expected datatype (tf.float32) and shape of the data.

Syntax to define a variable:
```
my_var = tf.Variable(initial_value=<can_be_a_tensor_or_number>
```
After defining the variables, we need to initialize them like this:
```
init = tf.global_variables_initializer()
sess.run(init)   // we didnt use "with tf.Session as sess" becuase we used interative session.
```
This way we initialize all variables.

For defining placeholders, we can easily say:
```
my_pholder = tf.placeholder(dtype, shape)   //     tf.placeholder(tf.float32, shape=(4,2))
```

sometimes we will see the expected shape depends on the data we will get from the final results. in this case we say: shape=(None, 2).

### TensorFlow Sample Neural Network:

we want to make a neural network for a simple linear regression WX + b = Z

(photo: basic_nn.jpg)

When generating random data in Nnumpy or TF, random uniform will accept min-max and shape:
```
rand_a = np.random.uniform(0,100,(5,5))  // will generate a 5x5 array between 0-100
```

TF element-wise multiplication:

```
tf.multiply(x,y)  returns x*y
```

__Important:__ when we use placeholders, we need to feed them. when we want to run the session with a placeholder:
```
results = sess.run(placeholder_name, feed_dict = {a: 10, b:12})
// a, b are the placeholders we defined beforehand.
```
The sample ANN we want to make has 10 features, and 3 dense neurons.

then we define the x, W, and b as placeholders and variables. then we calculate the formula as we see in the code.

Next step, is making an activation function (tanh, sigmoid, etc.) and pass the z which is the linear regression formula into the activation function.

As we see in the code, now we can calculate the results for 3 output neurons.

But in real world, we need to re-adjust W and b in order to optimize the results. For having this, we need to have a sort of cost function.

We now make a simple linear regression example.

#### Simple linear regression example:

we made some random values betwen 0-10 for x and y, and we added some noise to them and we see when we plot them there is a sort fo linear relationship between the points that we can predict.

- When we defined m, b variables, we will have to define the real y and predicted y (y_hat) and define the difference of these two values as the error. but for each set of x,y we will add up the errors.

- Next step, is to try to minimize the error. for this, we need to use gradiant descent.
```
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
```

Now we just need to minimize the error like this:
```
train = optimizer.minimize(error)
```
We can see the rest of the code in the notebook called TF-NN-part1.ipynb

#### TF Second Regression Sample:

- in Pandas, we can grab a subset of data, if its too big, like this:
```
sample = data.sample(n=20)  // returns 20 rows
```
<missed some text, not much to catch up. just check the code.>

- tf.reduce_sum()    It sums up the columns, or rows, or both.   

If it is columns we say   reduce_sum(array, 0),   For rows:  reduce_sum(array, 1),   and for both:  reduce_sum(array, [0,1])

#### TF Third Regression Sample using Estimator API:

There are several high level API's available for TensorFlow, such as Keras, Layers, Estimator, etc. Now we want to see how Estimator API works.

Estimator API has some different models that we can use:

1) tf.estimator.LinearClassifier() : constructs a linear classification model.
2) tf.estimator.LinearRegressor() : creates a linear regression model.
3) tf.estimator.DNNClassifier() : constructs a neural network classification model.
4) tf.estimator.DNNRegressor() : makes a neural network for linear regression.
5) tf.estimator.DNNLinearCombinedClassifier(): constructs a neural network and linear combined classifier.
6) tf.estimator.DNNLnearCombinedRegressor(): constructs a neural network and linear regression model.

Here is how we feed data into the API:
- first we define a list of feature columns. Later we learn how to use the built-in tf tools to deal with categorical columns so that you don't have to make dummy variables all the time.
- create the Estimator model.
- create a data input function. it can take the data as numpy array or pandas dataframe.
- call train, evaluate, and predict methods on the Estimator object, to train the model, evaluate it, and finally predict the labels for the test dataset.
##### Now let's see how we do it:
first we need to define featur columns:
```
feat_col = [tf.feature_column.numeric_column('x', shape=[1])]  // This makes all x column, into a list (because of the bracket we put around it)

```
Now we make the estimator:

```
estimator = tf.estimate.LinearRegressor(feature_column = feat_col)
```
Now we use sklearn train_test_split object to split our data.
For this we make 3 input functions one general, one for train, and one for test. as we see in the code.

Next step, is to train the estimator:
```
estimator.train(inpun_fn = input_func, steps=1000)
```
In fact there are easier and stronger ways of doing this, with better API's but still estimator API is a handy way for small projects.

#### TF First Classification Sample using Estimator API:

We import data (pima-indians-diabetes.csv) using pandas as a dataframe.

Then we normalize all numerical columns.

Now we create the feature_columns based on the dataset columns.

The next step, is to deal with categorical data. There are two ways of dealing with them:

1) __VOCABULARY LIST:__  

We can assign the value to the categorical column using vocabulary list:

```
a = tf.feature_column.categorical_column_with_vocabulary_list('column_name', ['A','B','C','D'])  
(A,B,C,D) are the possible values for the column.
```
2) __HASH BUCKET:__ When we have a big list of values in the categorical column, we can use hash bucket to automate this, instead of manually inputting the list of values.
```
a = tf.feature_column.categorical_column_with_hash_bucket('column_name', hash_bucket_size=100)
// 100 is the amount of values (distinct) we believe exist in the categorical column
```

##### Using TensorFlow to convert continuous values into categorical values:
We can see how Age has been categorized:
```
age_bucket = tf.feature_column.bucketized_column(age, boundaries=[20,30,40,50,60,70,80])  // 'age' is the name of the feature column we already made.
age = tf.feature_column.numeric_column('Age')
```
Now, it is time to make the input function. similar to what we did before.

Then we make the model. n_classes is the number of classes we have and want to predict. if it is binary, then we make it 2.

- Now let's see how we can use Dense Neural Networks to predict the classes. Here is how we define the model:

```
dnn_model = tf.estimator.DNNClassifier(hidden_units=[10,10,10])
10,10,10 means 3 layers, 3 nodes each. and Dense means all nodes are connected to all others from the next, previous layer.
```
When we want to train the model, we can't simply do it as we did before, to define the feature_columns and the steps. The reason is, because we had one of the columns as vocabulary list. we need to convert them to embedding.

```
embedded_group_col = tf.feature_column.embedding_column(assigned_group, dimension=4)  //4 is the number of the categories (A,B,C,D)
```
Now, we converted assigned_group to embedded_group_col, so we need to update the features_list and replace them in the list.
