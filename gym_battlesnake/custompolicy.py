import numpy as np
import tensorflow as tf

from stable_baselines.common.policies import ActorCriticPolicy, FeedForwardPolicy
from stable_baselines.a2c.utils import conv, linear, conv_to_fc

class CustomPolicy(FeedForwardPolicy):
    """
    FeedForwardPolicy but with a custom cnn feature extractor.
    """
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        ActorCriticPolicy.__init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=False)

        with tf.variable_scope("model", reuse=reuse):

            activ = tf.nn.elu
            # obs: (N, 11, 11, 6)
            
            # conv1: 3x3, stride 1, padding SAME -> (11x11x16)
            conv1 = activ(conv(self.processed_obs, 'c1', n_filters=16, filter_size=3, stride=1, init_scale=np.sqrt(2)))
            
            # conv2: 3x3, stride 1 -> (11x11x32)
            conv2 = activ(conv(conv1, 'c2', n_filters=32, filter_size=3, stride=1, init_scale=np.sqrt(2)))
            
            # conv3: 3x3, stride 2 -> (6x6x64)
            conv3 = activ(conv(conv2, 'c3', n_filters=64, filter_size=3, stride=2, init_scale=np.sqrt(2)))
            
            # Flatten -> Dense
            flat = conv_to_fc(conv3)
            
            # Fully connected layers
            fc1 = activ(linear(flat, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))
            pi_latent = vf_latent = activ(linear(fc1, 'fc2', n_hidden=256, init_scale=np.sqrt(2)))

            self.value_fn = linear(vf_latent, 'vf', 1)

            self.proba_distribution, self.policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self.initial_state = None
        self._setup_init()