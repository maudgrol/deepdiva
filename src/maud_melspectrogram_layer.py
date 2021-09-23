#!/usr/bin/env python
import tensorflow as tf


class LogMelSpectrogram(tf.keras.layers.Layer):
    """Compute log-magnitude mel-scaled spectrograms."""

    def __init__(self, sample_rate, fft_size, hop_size, n_mels,
                 f_min=None, f_max=None, **kwargs):
        super(LogMelSpectrogram, self).__init__(**kwargs) # inherits from tf.keras.layers.Layer
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.n_mels = n_mels
        self.f_min = f_min if f_min else 0.0
        self.f_max = f_max if f_max else sample_rate / 2
        self.mel_filterbank = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.n_mels,
            num_spectrogram_bins=self.fft_size // 2 + 1,
            sample_rate=self.sample_rate,
            lower_edge_hertz=self.f_min,
            upper_edge_hertz=self.f_max)

    def build(self, input_shape):
        self.non_trainable_weights.append(self.mel_filterbank)
        super(LogMelSpectrogram, self).build(input_shape)

    def call(self, waveforms):
        """
        Forward pass.
        :param waveforms: tf.Tensor, shape = (None, n_samples, 1)
            A Batch of mono waveforms.
        :return: log_mel_spectrograms : (tf.Tensor), shape = (None, freq, time, ch)
            The corresponding batch of log-mel-spectrograms
        """

        def _tf_log10(x):
            numerator = tf.math.log(x)
            denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
            return numerator / denominator

        def _power_to_db(magnitude, amin=1e-16, top_db=80.0):
            """
            https://librosa.github.io/librosa/generated/librosa.core.power_to_db.html
            """
            ref_value = tf.reduce_max(magnitude)
            log_spec = 10.0 * _tf_log10(tf.maximum(amin, magnitude))
            log_spec -= 10.0 * _tf_log10(tf.maximum(amin, ref_value))
            log_spec = tf.maximum(log_spec, tf.reduce_max(log_spec) - top_db)

            return log_spec

        waveforms = waveforms[:, :, 0]

        spectrograms = tf.signal.stft(waveforms,
                                      fft_length=self.fft_size,
                                      frame_length=self.fft_size,
                                      frame_step=self.hop_size,
                                      pad_end=True)

        magnitude_spectrograms = tf.abs(spectrograms)

        mel_spectrograms = tf.matmul(tf.square(magnitude_spectrograms),
                                     self.mel_filterbank)

        log_mel_spectrograms = _power_to_db(mel_spectrograms)

        # add channel dimension; change order of time and frequency dimensions
        log_mel_spectrograms = tf.expand_dims(log_mel_spectrograms, axis=-1)
        log_mel_spectrograms = tf.transpose(log_mel_spectrograms, perm=(0, 2, 1, 3))
        # flip the frequency axis
        log_mel_spectrograms = log_mel_spectrograms[:, ::-1, :, :]

        return log_mel_spectrograms

    def get_config(self):
        config = {
            'fft_size': self.fft_size,
            'hop_size': self.hop_size,
            'n_mels': self.n_mels,
            'sample_rate': self.sample_rate,
            'f_min': self.f_min,
            'f_max': self.f_max,
        }
        config.update(super(LogMelSpectrogram, self).get_config())

        return config