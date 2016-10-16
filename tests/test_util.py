import numpy as np
import tensorflow as tf
from homer.util import roll, _peaks

class RollTest(tf.test.TestCase):
  def test_roll(self):
    with self.test_session():
      assert np.array_equal(roll(tf.constant([1, 3, 5, 7, 9]), 1).eval(),
                            [0, 1, 3, 5, 7])
      assert np.array_equal(roll(tf.constant([1, 3, 5, 7, 9]), -1).eval(),
                            [3, 5, 7, 9, 0])
      assert np.array_equal(roll(tf.constant([1, 3, 5, 7, 9]), 3).eval(),
                            [0, 0, 0, 1, 3])
      assert np.array_equal(roll(tf.constant([1, 3, 5, 7, 9]), -3).eval(),
                            [7, 9, 0, 0, 0])
      assert np.array_equal(roll(tf.constant([1, 3, 5, 7, 9]), 5).eval(),
                            [0, 0, 0, 0, 0])
      assert np.array_equal(roll(tf.constant([1, 3, 5, 7, 9]), -5).eval(),
                            [0, 0, 0, 0, 0])
