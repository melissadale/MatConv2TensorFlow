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

    def maxpool2d(x, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                              padding='SAME')

    # Create model
    def conv_net(self, x, weights):
        x = tf.reshape(x, shape=[-1, self.input_size, self.input_size, 1])
        conv1 = self.conv2d(x, tf.reshape(weights['wc1'], shape=[5,5,1,20]))
        pool1 = self.maxpool2d(conv1)
        conv2 = self.conv2d(pool1, weights['wc2'])
        pool2 = self.maxpool2d(conv2)


        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.reshape(conv2, [-1, weights['wc3'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wc3']))
        fc1 = tf.nn.relu(fc1)
        # Apply Dropout
        # fc1 = tf.nn.dropout(fc1, dropout)

        # Output, class prediction
        out = tf.add(tf.matmul(fc1, weights['wc4']))
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