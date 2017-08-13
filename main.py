from tensortest import *

def import_data():
  # Load mnist training data
  from tensorflow.examples.tutorials.mnist import input_data
  return input_data.read_data_sets('MNIST_data', one_hot=True)

if __name__ == "__main__":
  with tf.Session() as sess:
    conv_net = ConvNetwork(sess)
    conv_net.build_graph()

    mnist = import_data()
    conv_net.train(mnist)
    conv_net.save('vars/model.ckpt')
  