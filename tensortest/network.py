import tensorflow as tf

class Network(object):
    """Class wrapping neural net graph and functions for mnist test"""
    
    def __init__(self, session):
      self.sess = session
      self.BATCH_SIZE = 100

    def build_graph(self):
      self.x = tf.placeholder(tf.float32, shape=[None, 784]) # inputs (flat 28 x 28 mnist)
      self.y_ = tf.placeholder(tf.float32, shape=[None, 10]) # correct outputs (nums 0-9)
      self.W = tf.Variable(tf.zeros([784,10])) # weights
      self.b = tf.Variable(tf.zeros([10])) # biases

      # Define regression model
      self.y = tf.matmul(self.x, self.W) + self.b

      # Define loss function
      self.cross_entropy = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y))

      self.train_step = tf.train.GradientDescentOptimizer(0.5).minimize(self.cross_entropy)

      # Init variables
      self.sess.run(tf.global_variables_initializer())

    def train(self, mnist):
      # Train network
      for _ in range(1000):
        batch = mnist.train.next_batch(self.BATCH_SIZE)
        self.train_step.run(feed_dict={self.x: batch[0], self.y_: batch[1]})

    def evaluate(self, mnist):
      correct_predictions = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
      print(accuracy.eval(feed_dict={self.x: mnist.test.images, self.y_: mnist.test.labels}))


