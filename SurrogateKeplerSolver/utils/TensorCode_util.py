import tensorflow as tf

import numpy as np
import sys
sys.path.append("..")
from TensorCode import MainCode as mct
from NormalCode import MainCode as mc

def do_step_wrapper(tau, n, m, rv):

    r, v = tf.split(rv, num_or_size_splits=2, axis=-1)
    r, v = mct.do_step(tau, n, m, r, v)
    return tf.concat([r, v], axis=1)


def do_step_wrapper_normal(tau, n, m, rv, gen_kepler_data = False, model = None):

    r, v = np.split(rv, indices_or_sections=2, axis=-1)#tf.split(rv, num_or_size_splits=2, axis=-1)
    r, v, data_for_kepler_model = mc.do_step(tau, n, m, r, v, gen_kepler_data=gen_kepler_data, model = model)
    return np.concatenate([r, v], axis=1), data_for_kepler_model

def do_step_wrapper_tfmodel(tau, n, m, rv, model):

    r, v = tf.split(rv, num_or_size_splits=2, axis=-1)
    r, v = mct.do_step_tfmodel(tau, n, m, r, v, model)
    return tf.concat([r, v], axis=1)

def execute_x_times_do_step_wrapper_tfmodel(tau, n, m, rv, num_of_steps_in_do_step, model):
    
    for _ in range(num_of_steps_in_do_step):
        rv = do_step_wrapper_tfmodel(tau, n, m, rv, model)
    return rv

def execute_x_times_do_step_wrapper(tau, n, m, rv, num_of_steps_in_do_step):
    
    for _ in range(num_of_steps_in_do_step):
        rv = do_step_wrapper(tau, n, m, rv)
    return rv