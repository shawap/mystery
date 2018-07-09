from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import os.path
import sys
import time
from six.moves import xrange  
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
##import mnist_read_data as input_data
#from tensorflow.examples.tutorials.mnist import mnist
import math
# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10
# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
import numpy as np # for preventing science notation 
import struct

def inference(images, hidden1_units, hidden2_units,hidden3_units):
  """Build the MNIST model up to where it may be used for inference.

  Args:
    images: Images placeholder, from inputs().
    hidden1_units: Size of the first hidden layer.
    hidden2_units: Size of the second hidden layer.
	......

  Returns:
    softmax_linear: Output tensor with the computed logits.
  """
  
  images = tf.cast(images, tf.float64)
  
  # Hidden 1
  with tf.name_scope('h1'):
    weights = tf.Variable(
        tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                            stddev=1.0 / math.sqrt(float(IMAGE_PIXELS)), dtype=tf.float64),
							name='weights')
    biases = tf.Variable(tf.zeros([hidden1_units], dtype=tf.float64),
                         name='biases')
    hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
  # Hidden 2
  with tf.name_scope('h2'):
    weights = tf.Variable(
        tf.truncated_normal([hidden1_units, hidden2_units],
                            stddev=1.0 / math.sqrt(float(hidden1_units)), dtype=tf.float64),
						    name='weights')
    biases = tf.Variable(tf.zeros([hidden2_units], dtype=tf.float64),
                         name='biases')
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
    # Hidden 3
  with tf.name_scope('h3'):
    weights = tf.Variable(
        tf.truncated_normal([hidden2_units, hidden3_units],
                            stddev=1.0 / math.sqrt(float(hidden2_units)), dtype=tf.float64),
						name='weights')
    biases = tf.Variable(tf.zeros([hidden3_units], dtype=tf.float64),
                         name='biases')
    hidden3 = tf.nn.relu(tf.matmul(hidden2, weights) + biases)
  # Linear
  with tf.name_scope('output'):
    weights = tf.Variable(
        tf.truncated_normal([hidden3_units, NUM_CLASSES],
                            stddev=1.0 / math.sqrt(float(hidden3_units)), dtype=tf.float64),
        name='weights')
    biases = tf.Variable(tf.zeros([NUM_CLASSES], dtype=tf.float64),
                         name='biases')
    logits = tf.matmul(hidden3, weights) + biases
  return logits


def loss_(logits, labels):
  """Calculates the loss from the logits and the labels.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].

  Returns:
    loss: Loss tensor of type float.
  """
  labels = tf.to_int64(labels)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='xentropy')
  return tf.reduce_mean(cross_entropy, name='xentropy_mean')


def training(loss, learning_rate):
  """Sets up the training Ops.
  Creates a summarizer to track the loss over time in TensorBoard.
  Creates an optimizer and applies the gradients to all trainable variables.
  The Op returned by this function is what must be passed to the
  `sess.run()` call to cause the model to train.

  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.

  Returns:
    train_op: The Op for training.
  """
  # Add a scalar summary for the snapshot loss.
  tf.summary.scalar('loss', loss)
  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  ##optimizer = tf.train.AdamOptimizer(0.0001)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op


def evaluation(logits, labels):
  """Evaluate the quality of the logits at predicting the label.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).

  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  # For a classifier model, we can use the in_top_k Op.
  # It returns a bool tensor with shape [batch_size] that is true for
  # the examples where the label is in the top k (here k=1)
  # of all logits for that example.
  ##############
  logits = tf.cast(logits, tf.float32)
  correct = tf.nn.in_top_k(logits, labels, 1)
  # Return the number of true entries.
  return tf.reduce_sum(tf.cast(correct, tf.int32))

# Basic model parameters as external flags.
FLAGS = None


def placeholder_inputs(batch_size):
  """Generate placeholder variables to represent the input tensors.

  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.

  Args:
    batch_size: The batch size will be baked into both placeholders.

  Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  # images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, mnist.IMAGE_PIXELS))
  #####images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, IMAGE_PIXELS))
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, IMAGE_PIXELS))
  labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
  return images_placeholder, labels_placeholder


def fill_feed_dict(data_set, images_pl, labels_pl):
  """Fills the feed_dict for training the given step.

  A feed_dict takes the form of:
  feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
  }

  Args:
    data_set: The set of images and labels, from input_data.read_data_sets()
    images_pl: The images placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().

  Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
  """
  # Create the feed_dict for the placeholders filled with the next
  # `batch size` examples.
  images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size,
                                                 FLAGS.fake_data)
  feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
  }
  return feed_dict


def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set):
  """Runs one evaluation against the full epoch of data.

  Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    images_placeholder: The images placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of images and labels to evaluate, from
      input_data.read_data_sets().
  """
  # And run one epoch of eval.
  true_count = 0  # Counts the number of correct predictions.
  steps_per_epoch = data_set.num_examples // FLAGS.batch_size
  num_examples = steps_per_epoch * FLAGS.batch_size
  for step in xrange(steps_per_epoch):
    feed_dict = fill_feed_dict(data_set,
                               images_placeholder,
                               labels_placeholder)
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
  precision = float(true_count) / num_examples
  print('Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))


def run_training():
  """Train MNIST for a number of steps."""
  # Get the sets of images and labels for training, validation, and
  # test on MNIST.
  data_sets = input_data.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data)

  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Generate placeholders for the images and labels.
    images_placeholder, labels_placeholder = placeholder_inputs(
        FLAGS.batch_size)

    # Build a Graph that computes predictions from the inference model.
    logits = inference(images_placeholder,
                             FLAGS.hidden1,
                             FLAGS.hidden2,
                             FLAGS.hidden3)
	
    # Add to the Graph the Ops for loss calculation.
    loss = loss_(logits, labels_placeholder)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = training(loss, FLAGS.learning_rate)

    # Add the Op to compare the logits to the labels during evaluation.
    eval_correct = evaluation(logits, labels_placeholder)

    # Build the summary Tensor based on the TF collection of Summaries.
    summary = tf.summary.merge_all()

    # Add the variable initializer Op.
    init = tf.global_variables_initializer()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

    # And then after everything is built:

    # Run the Op to initialize the variables.
    sess.run(init)

    
    saver = tf.train.Saver()


    # Start the training loop.
    for step in xrange(FLAGS.max_steps):
      start_time = time.time()

      # Fill a feed dictionary with the actual set of images and labels
      # for this particular training step.
      feed_dict = fill_feed_dict(data_sets.train,
                                 images_placeholder,
                                 labels_placeholder)

      # Run one step of the model.  The return values are the activations
      # from the `train_op` (which is discarded) and the `loss` Op.  To
      # inspect the values of your Ops or variables, you may include them
      # in the list passed to sess.run() and the value tensors will be
      # returned in the tuple from the call.
      _, loss_value = sess.run([train_op, loss],
                               feed_dict=feed_dict)

      duration = time.time() - start_time

      
      # Write the summaries and print an overview fairly often.
      if step % 100 == 0:
        # Print status to stdout.
        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
        # Update the events file.
        summary_str = sess.run(summary, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()

      # Save a checkpoint and evaluate the model periodically.
      if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        #checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
        save_path = saver.save(sess, "./ckpt/model.ckpt")
        
        print("Model saved in path: %s" % save_path)
        # Evaluate against the training set.
        print('Training Data Eval:')
        do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                data_sets.train)
        # Evaluate against the validation set.
        print('Validation Data Eval:')
        do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                data_sets.validation)
        # Evaluate against the test set.
        print('Test Data Eval:')
        do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                data_sets.test)
    #    print('epoch: %d' % data_sets.train.epochs_completed)

    print('____________________________________________________')
   # print(tf.global_variables())
    
    f = open('ignore1.txt', 'w')
    f_bin = open('parameter.txt', 'w')
    f_params = open('ignore2.txt', 'w')
    for v in tf.global_variables():
        #if 'nodes_' not in v.name:
        if v.name.find('nodes_') != 0:
            continue
		
        print(v.name)
        #f.write(v.name + ' ')
        f_params.write(v.name + ' ')
        shape = sess.run(tf.shape(v))
        print(shape)
        for item in shape.tolist():
            #f.write('%d ' % item)
            f_params.write('%d ' % item)
        f.write('\n')
        f_params.write('\n')
        value = sess.run(tf.transpose(v.value()))
        if len(shape) > 1:
            for v_ in value:
                #print(v_)
                for w in v_.tolist():
                    f.write('%24s ' % str(w))
					#########binary_value = struct.pack('f', w)
                    binary_value = struct.pack('d', w)
                    binary_str = ''.join('{:02x}'.format(x) for x in reversed(binary_value))
                    f_bin.write('%s ' % binary_str)
                f.write('\n')
                f_bin.write('\n')
            f_bin.write('\n')
        elif len(shape) == 1:
            #print(value)
            for v_ in value.tolist():
                f.write('%24s ' % str(v_))
				#############binary_value = struct.pack('f', v_)
                binary_value = struct.pack('d', v_)
                binary_str = ''.join('{:02x}'.format(x) for x in reversed(binary_value))
                f_bin.write('%s ' % binary_str)
            f.write('\n')
            f_bin.write('\n')
            f_bin.write('\n')
        else:
            f.write(str(value) + '\n')

    f.close()
    f_bin.close()
    f_params.close()

	
def main(_):
  np.set_printoptions(suppress=True) # for preventing science notation 
  
  with open("graph.proto", "wb") as file:
    graph = tf.get_default_graph().as_graph_def(add_shapes=True)
    file.write(graph.SerializeToString())

  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  run_training()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.15,
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--max_steps',
      type=int,
      default=30000,
      help='Number of steps to run trainer.'
  )
  parser.add_argument(
      '--hidden1',
      type=int,
      default=16,
      help='Number of units in hidden layer 1.'
  )
  parser.add_argument(
      '--hidden2',
      type=int,
      default=16,
      help='Number of units in hidden layer 2.'
  )
  parser.add_argument(
      '--hidden3',
      type=int,
      default=16,
      help='Number of units in hidden layer 3.'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=50,
      help='Batch size.  Must divide evenly into the dataset sizes.'
  )
  parser.add_argument(
      '--input_data_dir',
      type=str,
      default='/tmp/tensorflow/mnist/input_data',
      help='Directory to put the input data.'
  )
  parser.add_argument(
      '--log_dir',
      type=str,
      default='/tmp/tensorflow/mnist/logs/fully_connected_feed',
      help='Directory to put the log data.'
  )
  parser.add_argument(
      '--fake_data',
      default=False,
      help='If true, uses fake data for unit testing.',
      action='store_true'
  )

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
