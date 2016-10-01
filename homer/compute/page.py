import numpy as np
import tensorflow as tf

def load_png(filename_queue):
  reader = tf.WholeFileReader()
  key, value = reader.read(filename_queue)

  image = tf.image.decode_png(value, channels=1, dtype=tf.uint8)
  return image

def load_png_from_path(path):
  queue = tf.train.string_input_producer([path])
  return load_png(queue)
