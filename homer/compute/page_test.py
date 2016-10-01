import tensorflow as tf
from . import page

q = tf.train.string_input_producer(['samples/sonata.png'])

image = page.load_png(q)

with tf.Session() as sess:
  tf.initialize_all_variables().run()
  threads = tf.train.start_queue_runners()
  image_tensor = sess.run([image])
  print(image_tensor)
