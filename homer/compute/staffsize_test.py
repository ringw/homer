import tensorflow as tf
from . import *

q = tf.train.string_input_producer(['samples/sonata.png'])

with tf.Session() as sess:
  tf.initialize_all_variables().run()
  threads = tf.train.start_queue_runners()
  page = Page(q)
  print(sess.run([global_runhist(page)]))
