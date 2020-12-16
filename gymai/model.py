import numpy
import tensorflow
import datetime
from tensorflow import keras
from tensorflow.keras import layers, losses
import logging

log = logging.getLogger(__name__)

class ModelHolder(object):
    _ACTION_OUT = "action_out"
    _VALUE_OUT = "value_out"

    def __init__(self, input_shape, action_shape, name, act_coef = 2, vf_coef=1, ent_coef=0.005, batch_size=512, ppo_clip=0.2):
        self.name = name
        self.state_input = keras.Input(shape=input_shape)
        self.n_actions=action_shape
        self.ppo_clip = ppo_clip
        self.batch_size=batch_size
        #model_layers = self._dense_layers()
        model_layers = self._conv_layers()
        ent_reg = EntropyRegularizer(ent_coef)
        self._action_out = layers.Dense(
                action_shape,
                activation="softmax",
                name=self._ACTION_OUT,
                activity_regularizer=ent_reg)(model_layers)
        self._value_out = layers.Dense(
                1,
                activation="linear",
                bias_initializer=keras.initializers.RandomNormal(mean=-0.4),
                name=self._VALUE_OUT)(model_layers)

        self.model = keras.Model(
                inputs = self.state_input,
                outputs = [self._action_out, self._value_out],
                name = "a2c_agent")
        self.model.summary()
        self.model.add_metric(ent_reg.last_loss, name="entropy_loss")
        optimizer = keras.optimizers.Adam(lr=0.001)

        weights = {
                self._ACTION_OUT: act_coef,
                self._VALUE_OUT: vf_coef }
        losses = {
                self._ACTION_OUT: self._ppo_loss,
                self._VALUE_OUT: "mse"}
        self.model.compile(
                optimizer=optimizer,
                loss=losses,
                loss_weights=weights)


        self._epochs = 0
        log_dir = "logs/"+self.name
        writer = tensorflow.summary.create_file_writer(log_dir+"/game")
        self.reward_callback = RewardCallback(writer)
        self.tb_callback = keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=50,
                write_graph=True,
                update_freq="epoch",
                profile_batch=0,
                )

        self._cce = keras.losses.CategoricalCrossentropy()

    def _dense_layers(self):
        x = layers.Flatten()(self.state_input)
        x = layers.Dense(256, activation="relu", name="hidden_1")(x)
        x = layers.Dense(64, activation="relu", name="hidden_2")(x)
        return x

    def _conv_layers(self):
        x = self.state_input
        x = layers.Conv2D(64, 5, strides=2, activation="relu", name="conv_1", input_shape=x.shape[1:],padding="same")(x)
        x = layers.Conv2D(32, 3, strides=2, activation="relu", name="conv_2", padding="same")(x)
        x = layers.Conv2D(8, 3, activation="relu", name="conv_3", padding="same")(x)
        x = layers.Flatten()(x)
        return layers.Dense(32, activation="relu", name="dense_1")(x)

    def _action_loss(self, y_true, y_pred):
        advantages = y_true[:, :1]
        actions_taken = y_true[:, 1:self.n_actions+1]
        return self._cce(actions, y_pred, sample_weight=advantages)

    def _ppo_loss(self, y_true, y_pred):
        advantages = y_true[:, :1]
        actions_taken = y_true[:, 1:self.n_actions+1]
        action_pred = y_true[:, 1+self.n_actions:]

        action_prob = y_pred * actions_taken
        old_action_prob = action_pred * actions_taken
        r = action_prob/(old_action_prob + 1e-10)
        p1 = r * advantages
        p2 = keras.backend.clip(r, min_value=1 - self.ppo_clip, max_value=1 + self.ppo_clip) * advantages
        loss = -keras.backend.mean(keras.backend.minimum(p1, p2))
        return loss

    def _periodic_save(self):
        if self._epochs % 10 == 0:
            log.info("saving model")
            path = "models/"+self.name+".h5"
            self.model.save(path)

    def train(self, states, discounted_rewards, advantages, actions_taken, actions_pred):
        action_y = numpy.hstack([advantages, actions_taken, actions_pred])
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
                callbacks=[self.tb_callback])
        self._epochs += 1
        self._periodic_save()

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

class RewardCallback(object):
    def __init__(self, writer):
        self.writer = writer
        self.total_rewards = 0
        self.last_rewards = 0

    def report_game(self, game_num, rewards):
        self.total_rewards += rewards
        with self.writer.as_default():
            tensorflow.summary.scalar('rewards_total', data=self.total_rewards, step=game_num)
            tensorflow.summary.scalar('rewards_last', data=rewards, step=game_num)
            self.writer.flush()
