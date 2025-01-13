import keras.api
import tensorflow as tf
import keras
from utils import TensorCode_util as tc


class PlanetMassModel_WithKeplerModel(keras.Model):
    def __init__(self, time_step, initial_masses, num_of_steps, trained_model, mass_penalty=0.0, **kwargs):
        super().__init__()
        self.time_step = tf.constant(time_step, dtype=tf.float64)
        self.num_of_steps = tf.constant(num_of_steps, dtype=tf.int64)
        self._masses = keras.Variable(
            initial_masses, trainable=True, dtype=tf.float64)
        self.n = self._masses.shape[0]
        self._mass_penalty = tf.cast(mass_penalty, dtype=tf.float64)
        self.switch = True
        self.trained_model = trained_model
        self._mask = tf.constant(kwargs["mask"], dtype=tf.float64)

    @tf.function
    def train_step(self, data):

        rv0, rv1 = tf.split(data, num_or_size_splits=2, axis=-1)

        rv0 = tf.cast(rv0, dtype=tf.float64)
        rv1 = tf.cast(rv1, dtype=tf.float64)

        with tf.GradientTape() as tape:
            result = tf.map_fn(
                lambda rv: tc.execute_x_times_do_step_wrapper_tfmodel(
                    rv=rv, tau=self.time_step, n=self.n, m=self._masses, num_of_steps_in_do_step=self.num_of_steps, model=self.trained_model),
                rv0,
                fn_output_signature=tf.TensorSpec(
                    shape=(None, 6), dtype=tf.float64),
            )  # Forward pass
            # Compute the loss value
            loss = self.loss_fn(rv1, result)

        # Compute gradients
        gradients = tf.multiply(tape.gradient(loss, self._masses), self._mask)
        # Update weights
        self.optimizer.apply_gradients([(gradients, self._masses)])
        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(rv1, result)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def loss_fn(self, y_true, y_pred):
        r1, v1 = tf.split(y_true, num_or_size_splits=2, axis=-1)
        r2, v2 = tf.split(y_pred, num_or_size_splits=2, axis=-1)

        main_loss = tf.cast(tf.reduce_sum(tf.pow(r1 - r2, 2)) +
                            tf.reduce_sum(tf.pow(v1 - v2, 2)), dtype=tf.float64)

        # penalty = sum( squared(max(-m, 0)) ) * (multiplier=1)
        mass_penalty = self._mass_penalty * \
            tf.reduce_sum(tf.square(tf.maximum(tf.multiply(
                tf.cast(-1, dtype=tf.float64), self._masses), 0.0)))

        loss = main_loss + mass_penalty
        return loss

    @property
    def masses(self):
        return (self._masses).numpy()


class PrintMasses(keras.callbacks.Callback):

    def __init__(self, model):
        super(PrintMasses, self).__init__()
        self._model = model

    def on_epoch_end(self, epoch, logs=None):
        print(f"masses = {self._model.masses}")


class RecordHistory(keras.callbacks.Callback):

    def __init__(self, model):
        super(RecordHistory, self).__init__()
        self._model = model
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        self.history.append(self._model._masses.numpy())
