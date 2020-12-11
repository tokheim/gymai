import numpy
import tensorflow
import datetime
from tensorflow import keras
from tensorflow.keras import layers, losses

class ModelHolder(object):
    _ACTION_OUT = "action_out"
    _VALUE_OUT = "value_out"

    def __init__(self, input_shape, action_shape, vf_coef=0.5, ent_coef=0.001, batch_size=128):
        self.state_input = keras.Input(shape=input_shape)
        self.batch_size=batch_size
        model_layers = self._model_layers()
        ent_reg = EntropyRegularizer(ent_coef)
        self._action_out = layers.Dense(
                action_shape,
                activation="softmax",
                name=self._ACTION_OUT,
                activity_regularizer=ent_reg)(model_layers)
        self._value_out = layers.Dense(
                1,
                activation="linear",
                name=self._VALUE_OUT)(model_layers)

        self.model = keras.Model(
                inputs = self.state_input,
                outputs = [self._action_out, self._value_out],
                name = "a2c_agent")
        self.model.summary()
        self.model.add_metric(ent_reg.last_loss, name="entropy_loss")
        optimizer = keras.optimizers.Adam(lr=0.001)

        weights = {
                self._ACTION_OUT: 1,
                self._VALUE_OUT: vf_coef }
        losses = {
                self._ACTION_OUT: self._action_loss,
                self._VALUE_OUT: "mse"}
        self.model.compile(
                optimizer=optimizer,
                loss=losses,
                loss_weights=weights)


        self._epochs = 0
        log_dir = "logs/run_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = tensorflow.summary.create_file_writer(log_dir+"/game")
        self.reward_callback = RewardCallback(writer)
        self.tb_callback = keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                write_graph=True,
                write_images=True,
                update_freq="epoch",
                profile_batch=0,
                )

        self._cce = keras.losses.CategoricalCrossentropy()

    def _model_layers(self):
        x = layers.Flatten()(self.state_input)
        x = layers.Dense(256, activation="relu", name="hidden_1")(x)
        x = layers.Dense(64, activation="relu", name="hidden_2")(x)
        return x

    def _action_loss(self, y_true, y_pred):
        advantages = y_true[:, :1]
        actions = y_true[:, 1:]
        return self._cce(actions, y_pred, sample_weight=advantages)

    def train(self, states, discounted_rewards, advantages, actions_taken):
        action_y = numpy.hstack([advantages, actions_taken])
        batch_size = min(states.shape[0], self.batch_size)
        y = {
                self._VALUE_OUT : discounted_rewards,
                self._ACTION_OUT : action_y }

        self.model.fit(
                x=states,
                y=y,
                epochs=self._epochs+1,
                batch_size=self.batch_size,
                shuffle=True,
                initial_epoch=self._epochs,
                callbacks=[self.reward_callback, self.tb_callback])
        self._epochs += 1

    def apply(self, state):
        data = self.model.predict(state)
        return data[0][0], data[1][0]



class EntropyRegularizer(keras.regularizers.Regularizer):
    def __init__(self, strength):
        super(EntropyRegularizer, self).__init__()
        self.strength = strength
        self._last_loss = 0

    def __call__(self, x):
        #minus entropy. Since model seeks to minimize loss, this will
        #move probs to even values where entropy is highest
        losses = tensorflow.reduce_sum(x * tensorflow.math.log(x+1e-10), 1)
        self._last_loss = tensorflow.reduce_mean(losses)
        return self.strength * tensorflow.reduce_sum(losses)

    @property
    def last_loss(self):
        return self._last_loss

    def get_config(self):
        return { "strength": self.strength }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class RewardCallback(keras.callbacks.Callback):
    def __init__(self, writer):
        self.writer = writer
        self.total_rewards = 0
        self.last_rewards = 0

    def add_rewards(self, rewards):
        self.total_rewards += rewards
        self.last_rewards = rewards

    def on_epoch_end(self, epoch, logs=None):
        with self.writer.as_default():
            tensorflow.summary.scalar('rewards_total', data=self.total_rewards, step=epoch)
            tensorflow.summary.scalar('rewards_last', data=self.last_rewards, step=epoch)
            self.writer.flush()
