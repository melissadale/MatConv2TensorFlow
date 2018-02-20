import tensorflow as tf
import pickle


class cnn():
    # """
    # Modified from
    # https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network_raw.py
    # """
    input_size = 28 # e.g. 28x28 input

    def __init__(self, test_set):
        self.test_set = test_set
        self.model = pickle.load(open( "model.p", "rb" ) )
        self.learning_rate = self.model[0]['lr']
        self.weights = {
            'wc1': tf.Variable(self.model[0]['weights']),
            'wc2': tf.Variable(self.model[2]['weights']),
            'wc3': tf.Variable(self.model[4]['weights']),
            'wc4': tf.Variable(self.model[6]['weights'])
        }

        self.biases = {
            'bc1': tf.Variable(self.model[0]['bias']),
            'bc2': tf.Variable(self.model[2]['bias']),
            'bc3': tf.Variable(self.model[4]['bias']),
            'bc4': tf.Variable(self.model[6]['bias']),
        }


    # tf Graph input
    X = tf.placeholder(tf.float32, [None, input_size, input_size])

    # Create some wrappers for simplicity
    def conv2d(self, x, W, b, strides=1):
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    # Create model
    def conv_net(self, x, weights, biases):
        x = tf.reshape(x, shape=[-1, self.input_size, self.input_size, 1])
        conv1 = self.conv2d(x, tf.reshape(weights['wc1'], [4, 4, 1, 60]) , biases['bc1']) # TODO
        conv2 = self.conv2d(conv1, weights['wc2'], biases['bc2'])
        conv3 = self.conv2d(conv2, weights['wc3'], biases['bc3'])

        # Fully connected layer
        fc1 = tf.reshape(conv3, [-1, weights['wc4'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wc4']), biases['bc4'])
        out = tf.nn.relu(fc1)
        return out

    def run(self):
        these = []
        # Construct model
        logits = self.conv_net(self.X, self.weights, self.biases)
        prediction = tf.argmax(tf.nn.softmax(logits), 1)

        feed_dict = {self.X: self.test_set}
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            classification = sess.run(prediction, feed_dict)
            these.append(classification)

        return these

