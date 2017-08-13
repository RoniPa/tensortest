import tensorflow as tf
import argparse
from tensortest import *

def import_data():
  # Load mnist training data
  from tensorflow.examples.tutorials.mnist import input_data
  return input_data.read_data_sets('MNIST_data', one_hot=True)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-s', '--save', help='persist model after training (saved to vars/model.ckpt)',
                      action='store_true')
  args = parser.parse_args()

  with tf.Session() as sess:
    conv_net = ConvNetwork(sess)
    conv_net.build_graph()

    mnist = import_data()
    conv_net.train(mnist)

    if args.save:
      conv_net.save('vars/model.ckpt')
  