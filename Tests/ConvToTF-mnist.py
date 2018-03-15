import tensorflow as tf
import pickle
import numpy as np

class cnn():
    # """
    # Modified from
    # https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network_raw.py
    # """
    input_size = 28  # e.g. 28x28 input

    def __init__(self, test_set):
        self.test_set = test_set
        self.model = pickle.load(open("model.p", "rb" ) )
        # self.learning_rate = self.model[0]['lr']
        self.weights = {
            'wc1': tf.Variable(self.model[0]['weights']),
            'wc2': tf.Variable(self.model[2]['weights']),
            'wc3': tf.Variable(self.model[4]['weights']),
            'wc4': tf.Variable(self.model[5]['weights'])
        }

    # tf Graph input
    X = tf.placeholder(tf.float32, [None, input_size, input_size])

    # Create some wrappers for simplicity
    def conv2d(self, x, W, strides=2):
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        return tf.nn.relu(x)

    # Create model
    def conv_net(self, x, weights):
        x = tf.reshape(x, shape=[-1, self.input_size, self.input_size, 1])
        conv1 = self.conv2d(x, tf.reshape(weights['wc1'], shape=[5,5,1,20]))
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=self.model[1]['pool'], strides=self.model[1]['stride'])
        conv2 = self.conv2d(pool1, weights['wc2'])
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=self.model[3]['pool'], strides=self.model[3]['stride'])

        # conv3 = self.conv2d(pool2, weights['wc3'])
        dense = tf.layers.dense(pool2, weights['wc3'], activation=tf.nn.relu)

        # Fully connected layer
        fc1 = tf.reshape(dense, [-1, weights['wc4'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wc4']))
        out = tf.nn.relu(fc1)
        return out

    def run(self):
        these = []
        # Construct model
        logits = self.conv_net(self.X, self.weights)
        prediction = tf.argmax(tf.nn.softmax(logits), 1)

        feed_dict = {self.X: self.test_set}
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            classification = sess.run(prediction, feed_dict)
            these.append(classification)

        return these



def main(unused_argv):
  # # Load training and eval data
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  eval_data = mnist.test.images  # Returns np.array
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

  predicted = cnn(eval_data)
  positive = predicted.run()

if __name__ == "__main__":
  tf.app.run()