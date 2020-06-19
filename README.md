# test_repo

# Using tensorflow on MNIST data

created_via API call

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot = True)

# lets see the images
import numpy as np
from matplotlib import pyplot as plt
first_image = mnist.train.images[412]
first_image = np.array(first_image, dtype = "float")
first_image = first_image.reshape((28,28))
plt.imshow(first_image)
plt.show()

with tf.Session() as sess:
    print(tf.random_normal([784,256]).eval())
    
# h1 = 256
# h2 = 256
# random weights
n_input = 784
n_hidden_1 = 256
n_hidden_2 = 256
n_classes = 10

weights = {"h1" : tf.Variable(tf.random_normal([n_input,n_hidden_1])),
           "h2" : tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
           "out": tf.Variable(tf.random_normal([n_hidden_2,n_classes]))}

biases =  {"h1" : tf.Variable(tf.random_normal([n_hidden_1])),
           "h2" : tf.Variable(tf.random_normal([n_hidden_2])),
           "out": tf.Variable(tf.random_normal([n_classes]))}


# forward propogation
def forward_prop(x,weights,biases):
    in_layer1 = tf.add(tf.matmul(x,weights['h1']),biases['h1'])
    out_layer1 = tf.nn.relu(in_layer1)
    
    in_layer2 = tf.add(tf.matmul(out_layer1,weights['h2']),biases['h2'])
    out_layer2 = tf.nn.relu(in_layer2)
    
    output = tf.add(tf.matmul(out_layer2,weights['out']),biases['out'])
    return output
 
 

x = tf.placeholder('float',[None, n_input])
y = tf.placeholder(tf.int32,[None, n_classes])
pred = forward_prop(x,weights,biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred,labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate = 0.01)
optimize = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

c , _ = sess.run([cost,optimize], feed_dict = {x:mnist.train.images, y: mnist.train.labels})

# optimizer is changing this trainable data 

tf.trainable_variables()

# run optimizer multiple times
for i in range(25):
    c , _ = sess.run([cost,optimize], feed_dict = {x:mnist.train.images, y: mnist.train.labels})
    print(c)
    
# how many predictions are correct    
predictions = tf.argmax(pred, 1)
true_labels = tf.argmax(y,1)
correct_predictions = tf.equal(predictions,true_labels)
predictions_eval, labels, correct_pred = sess.run([predictions ,true_labels,correct_predictions] , feed_dict= {x:mnist.train.images, y : mnist.train.labels})
correct_pred.sum()






    
    
