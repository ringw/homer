import tensorflow as tf
from homer.compute import page
from homer.compute import staffsize

class StaffSizeTest(tf.test.TestCase):
  def test_global_runhist(
      self, file='../samples/sonata.png', staffspace=18, staffthick=3):
    """Tests the global run count again known values for a page."""
    with self.test_session() as sess:
      p = page.load_png_from_path(file)
      tf.train.start_queue_runners(sess=sess)
      white_hist, black_hist = sess.run(staffsize.global_runhist(p))
      assert white_hist.argmax() == staffspace
      assert black_hist.argmax() == staffthick

  def test_single_staffsize(
      self, file='../samples/sonata.png', staffspace=18, staffthick=3):
    """Tests the global run count again known values for a page."""
    with self.test_session() as sess:
      p = page.load_png_from_path(file)
      tf.train.start_queue_runners(sess=sess)
      staffsize.get_staffsize(p)
      assert sess.run([p.staff_space, p.staff_thick, p.staff_dist]) == \
          [staffspace, staffthick, staffspace + staffthick]

  def test_ossia_single_staffsize(self, file='../samples/trio.png'):
    with self.test_session() as sess:
      p = page.load_png_from_path(file)
      tf.train.start_queue_runners(sess=sess)
      assert sess.run([staffsize.get_staffsize(p)]) == [-1]
