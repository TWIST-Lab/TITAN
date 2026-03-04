import tensorflow as tf
from sionna.phy.utils import insert_dims
from sionna.sys import PFSchedulerSUMIMO

class CoarsePFSchedulerSUMIMO(PFSchedulerSUMIMO):
    """Proportional fairness scheduler operating on resource blocks.

    This scheduler groups ``rb_size`` consecutive subcarriers into a single
    resource block. Scheduling decisions are taken per block and replicated
    to all subcarriers inside the block.
    """

    def __init__(self, *args, rb_size=12, **kwargs):
        super().__init__(*args, **kwargs)
        self._rb_size = int(rb_size)
        self._num_rbs = self._num_freq_res // self._rb_size

    def call(self, rate_last_slot, rate_achievable_curr_slot):
        tf.debugging.assert_equal(
            rate_last_slot.shape,
            self._batch_size + [self._num_ut],
            message="Inconsistent 'rate_last_slot' shape")

        tf.debugging.assert_equal(
            rate_achievable_curr_slot.shape,
            self._batch_size + [self._num_ofdm_sym, self._num_freq_res, self._num_ut],
            message="Inconsistent 'rate_achievable_curr_slot' shape")

        rate_last_slot = tf.cast(rate_last_slot, self.rdtype)
        rate_achievable_curr_slot = tf.cast(rate_achievable_curr_slot, self.rdtype)

        # Update time-averaged achieved rate
        self._rate_achieved_past.assign(
            self.beta * self._rate_achieved_past +
            (1 - self.beta) * rate_last_slot)
        rate_achieved_past = insert_dims(self._rate_achieved_past, 2, axis=-2)

        # Average achievable rate over RBs
        new_shape = self._batch_size + [self._num_ofdm_sym,
                                        self._num_rbs, self._rb_size,
                                        self._num_ut]
        rate_rb = tf.reshape(rate_achievable_curr_slot, new_shape)
        rate_rb = tf.reduce_mean(rate_rb, axis=-2)

        # Compute PF metric per RB
        pf_metric_rb = rate_rb / rate_achieved_past

        # Store PF metric expanded to subcarriers
        self._pf_metric.assign(tf.repeat(pf_metric_rb, self._rb_size, axis=2))

        # Schedule users per RB
        scheduled_ut_rb = tf.argmax(pf_metric_rb, axis=-1)
        is_scheduled_rb = tf.one_hot(scheduled_ut_rb, depth=self._num_ut)
        is_scheduled_rb = tf.expand_dims(is_scheduled_rb, axis=-1)
        is_scheduled_rb = tf.tile(is_scheduled_rb,
                                  [1]*(3 + len(self._batch_size)) +
                                  [self._num_streams_per_ut])

        # Replicate allocation to all subcarriers of the RB
        is_scheduled = tf.repeat(is_scheduled_rb, self._rb_size, axis=2)
        return tf.cast(is_scheduled, tf.bool)
