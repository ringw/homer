from homer.page import Page
from homer.rotate import get_rotated_page
from homer.scale import get_scaled_page
from homer.scale import SCALED_STAFFSIZE
from homer.compute.staffsize import get_staffsize
import tensorflow as tf

def page_queue(page_path, image, capacity=10):
  page = create_page(image)
  queue = tf.FIFOQueue(capacity, dtypes=[tf.string, tf.float32])
  enqueue_op = tf.cond(
      can_process(page), lambda: queue.enqueue((page_path, page.image)), lambda: tf.no_op())
  qr = tf.train.QueueRunner(queue, [enqueue_op])
  return queue, qr

def create_page(image):
  page = Page(image)
  page = get_rotated_page(page)
  page = get_scaled_page(page)
  return page

def can_process(page):
  return abs(get_staffsize(page) - SCALED_STAFFSIZE) <= 1
