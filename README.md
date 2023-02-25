# Neural-Networks-with-TensorFlow - Deep Learning

Date: 23/02/2023

Just some random helpful points taken during this learning...

**Reference:**

[The Ultimate Guide to Artificial Neural Networks (ANN) - Blogs - SuperDataScience | Machine Learning | AI | Data Science Career | Analytics | Success](https://www.superdatascience.com/blogs/the-ultimate-guide-to-artificial-neural-networks-ann)

<div id='center'> . . . </div>

**What is an ANN? —** Artificial Neural Network: An artificial neuron network (**ANN**) is a computational model based on the structure and functions of biological neural networks/brain.

**What is a Neuron? —** The ****neuron that forms the basis of all Neural Networks is an imitation of what has been observed in the human brain.

**What is the activation function? — it** is the process applied to the weighted input value once it enters the neuron.

The activation function is something of a mysterious ingredient added to the input ingredients already bubbling in the **neuron’s** pot. If weighted **input values** are shampoo, floor polish, and gin, the neuron would be the deep black pot. The activation function is the open flame beneath that congeals the concoction into something new, the **output value.**

The activation function decides whether a neuron should be activated or not by calculating the weighted sum and further adding bias to it. The purpose of the activation function is to introduce **non-linearity** into the output of a neuron.

Nonlinearity is a statistical term that describes the relationship between dependent and independent variables. It describes a link that cannot be expressed with a straight line.

In a nonlinear relationship, a change in either of the inputs does not reflect a corresponding change in the output.
.

It is important to remember; you must either **standardize** the values of your independent variables or **normalize** them. These processes keep your variables within a similar range, so it is easier for your Neural Network to process them. This is essential for the operational capacity of your Neural Network.

Outputs in a Neural Net can be either:

- continuous (price)
- binary (yes or no)
- or categorical (multiple variables).

**Weights are a pivotal factor in a Neural Network’s functioning**.

Weights are how Neural Networks learn.

Based on each weight, the Neural Network decides what information is important and what isn’t.

The **weights** are what you will adjust through the process of learning. When you are training your Neural Network, not unlike with your body, the work is done with weights.

## Types of activation functions

- **The Threshold Function —** If the weighted sum is valued as less than 0, the TF will pass on the value 0. If the value is equal to or more than 0, the TF passes on 1. It is a yes or no, black or white, binary function.
    
    
- **The Sigmoid Function —** Sigmoid’s curvature means it is far better suited to probabilities when applied at the output layer of your NN**.**
    
    **Uses:** Usually used in the **output layer** of a binary classification, where the result is either 0 or 1, as the value for the sigmoid function lies between 0 and 1 only, so the result can be predicted easily to be ***1*** if the value is greater than **0.5** and ***0*** otherwise.
    
- **Hyperbolic Tangent Function —** The activation that works almost always better than the sigmoid function is the Tanh function, also known as the **Tangent Hyperbolic function**. It’s actually a mathematically shifted version of the sigmoid function. Both are similar and can be derived from each other.
    
    It is willing to delve deep below the x-axis and its 0 value to the icy pits of the lowest circle, where the value -1 slumbers.
    
    **Uses:** Usually used in hidden layers of a neural network as its values lie between **-1 to 1** 
    hence the mean for the hidden layer comes out to be 0 or very close to it, hence helping in *centering the data* by bringing the mean close to 0. This makes learning for the next layer much easier.
    
- ****RELU Function —**** *Rectified linear unit*. It is the most widely used activation function. Chiefly implemented in *hidden layers* of the Neural network.
    
    ReLu is less computationally expensive than tanh and sigmoid because it involves simpler mathematical operations. At a time, only a few neurons are activated, making the network sparse and making it efficient and easy for computation.
    
    In simple words, RELU learns *much faster* than the sigmoid and Tanh functions.
    
- ****SoftMax Function —**** The SoftMax function is also a type of sigmoid function but is handy when we are trying to handle multi-class classification problems.
    
    **Uses:** Usually used when trying to handle multiple classes. The softmax function was commonly found in the output layer of image classification problems. The softmax function would squeeze the outputs for each class between 0 and 1 and would also divide by the sum of the outputs.
    
    - The softmax function is ideally used in the output layer of the classifier, where we are actually trying to attain the probabilities to define the class of each input.
    - The basic rule of thumb is if you really don’t know what activation function to use, then simply use *RELU,* as it is a general activation function in hidden layers and is used in most cases these days.
    - If your output is for binary classification, then the *sigmoid function* is a very natural choice for the output layer.
    - If your output is for multi-class classification, then, Softmax is very useful for predicting the probabilities of each class.
    

### Biase and Weights in NN([Effect of Bias in Neural Network - GeeksforGeeks](https://www.geeksforgeeks.org/effect-of-bias-in-neural-network/))

In a Neural network, some inputs are provided to an artificial neuron, and with each input a weight is associated. Weight increases the steepness of activation function. This means weight decide how fast the activation function will trigger whereas bias is used to delay the triggering of the activation function.

Process in a Neuron:

**`output  =  sum (weights * inputs) + bias`**

![Bias](https://media.geeksforgeeks.org/wp-content/uploads/neuron.png)

Increasing the weight, the steepness increases.

Therefore, it can be inferred that: **The more the weight earlier the activation function will trigger**.

`**Bias** helps in controlling the value at which the activation function will trigger.`

# Why do we need a Non-linear activation function?

A neural network without an activation function is essentially a linear regression model. The activation function does the non-linear transformation to the input, making it capable of learning and performing more complex tasks.

## What is a Cost function?

A cost function is a difference between the predicted and the actual value. Mostly half of the squared difference between the predicted and actual them. The closer the better.

- Loss function — The lower, the better the model. `Gradient decent`.
- Reward function — The higher, the better the model.

## How to adjust the weight to reduce the cost function

- ***Brute-force approach*** — This is far better suited to a single-layer feed-forward network. Here you take a number of possible weights.
    
    ![Brite-force](https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/45_blog_image_26.png)
    
    You could find your way to the best weight through a simple process of elimination. You could trial every weight through your network and, one by one gets closer and closer to your optimal weight. On a simple level, this would suffice, say, if you only had a single weight to optimize. But the larger a network becomes, the number of weights that will emerge means this method is **impracticable**.
    
- ***[Gradient Descent](https://iamtrask.github.io/2015/07/27/python-network-part2/)*** — Instead of going through every weight one at a time and ticking every wrong weight off as you go, **you look at the angle of the cost function line**.
    
    If the slope is negative, like it is from the highest red dot on the above line, that means you must go downhill from there.
    
    ![Gradient Discent](https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/45_blog_image_29.png)
    

This eliminates a vast number of incorrect weights on the way down. It also reduces the time and effort spent on finding the right weight.

**Stochastic Gradient Descent —** Better than the Gradient Descent. It moves each row after the other adjusting the weights each time.

## What is backpropagation?

This is where we feed the end data back through the Neural Network and then adjust the weighted synapses between the input value and the neuron. This helps reduce the cost function.

**More to check!!!:** 

- Batch Learning
- More on Backpropagation
