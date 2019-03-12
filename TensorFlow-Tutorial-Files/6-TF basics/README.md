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






