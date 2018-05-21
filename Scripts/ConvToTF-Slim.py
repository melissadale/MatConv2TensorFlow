import tensorflow as tf
from tensorflow.contrib import slim
import pickle
import numpy as np

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./MNIST-data/", one_hot=True)


input_size = 28  # e.g. 28x28 input


model = pickle.load(open("./Tests/model.p", "rb"))
learning_rate = model[0]['lr']

weights = {
    'wc1': model[0]['weights'],  # 5 x 5 x 20
    'wc2': model[2]['weights'],  # 5 x 5 x 20 x 50
    'wc3': model[4]['weights'],  # 4 x 4 x 50 x 500
    'out': model[5]['weights']   # 500 x 10
}

biases = {
    'bc1': model[0]['bias'],  # 20
    'bc2': model[2]['bias'],  # 50
    'bc3': model[4]['bias'],  # 500
    'out': model[5]['bias']  # 10
}

# https://github.com/initialized/tensorflow-tutorial/blob/master/mnist-slim/MNIST%20Slim.ipynb
def cnn(inputs):
   with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu):

        # First Group: Convolution + Pooling 28x28x1 => 28x28x20 => 14x14x20
        net = slim.conv2d(inputs, 20, [5, 5], padding='SAME', scope='conv1')  # 28x28x20
        net = slim.max_pool2d(net, [2, 2], scope='pool1')  # 14x14x20

        # Second Group: Convolution + Pooling 14x14x20 => 10x10x40 => 5x5x40
        net = slim.conv2d(net, 50, [5, 5], padding='SAME', scope='conv2')  # 14 x 14 x 50
        net = slim.max_pool2d(net, [2, 2], scope='pool2')  # 7 x 7 x 50

        net = slim.conv2d(net, 500, [4, 4], padding='SAME', scope='conv3')
        # Reshape: 5x5x40 => 1000x1
        net = tf.reshape(net, [-1, 7*7*500])  # 800

        # Fully Connected Layer: 1000x1 => 1000x1 - 800
        net = slim.fully_connected(net, 500, scope='fc')

        # Output Layer: 1000x1 => 10x1
        net = slim.fully_connected(net, 10, scope='out')

   return net


# Create the placeholder tensors for the input images (x), the training labels (y_actual)
# and whether or not dropout is active (is_training)
x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='Inputs')
y_actual = tf.placeholder(tf.float32, shape=[None, 10], name='Labels')


logits = cnn(x)
prediction = tf.nn.softmax(logits)

# # Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_actual, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# https://stackoverflow.com/questions/43816575/how-can-i-i-initialize-the-weights-in-slim-conv2d-with-the-value-of-existing-m?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
assign_op, feed_dict_init = slim.assign_from_values({
    'conv1/weights': weights['wc1'],
    'conv2/weights': weights['wc2'],
    'conv3/weights': weights['wc3'],
    'out/weights': weights['out'],
    #
    'conv1/biases': biases['bc1'],
    'conv2/biases': biases['bc2'],
    'conv3/biases': biases['bc3'],
    'out/biases': biases['out']
})


def InitAssignFn(sess):
    sess.run(assign_op, feed_dict_init)


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    sess.run(assign_op, feed_dict_init)

    # Calculate accuracy for 256 MNIST test images
    print("Testing Accuracy:", \
          sess.run(accuracy, {x: np.reshape(mnist.test.images[:256], (-1, 28, 28, 1)),  y_actual: mnist.test.labels[:256]}))
