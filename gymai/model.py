import numpy
import tensorflow
import datetime
import random
import math
import tensorflow
from tensorflow import keras
from tensorflow.keras import layers, losses
import logging

log = logging.getLogger(__name__)

def create_model(conf, input_shape, action_shape, train_planner, name, start_epoch=0):
    state_input = keras.Input(shape=input_shape)
    layers = _gen_layers(conf.get("type"), state_input)
    ent_reg = EntropyRegularizer(conf["ent_coef"], action_shape)
    weights = {
            ModelHolder._ACTION_OUT: conf["act_coef"],
            ModelHolder._VALUE_OUT: conf["vf_coef"]
    }
    steps = conf.get("train_steps", 1e9)
    lr = conf["lr"]
    lr_scheduler = LinearScheduler(
        steps=steps,
        start=lr,
        end=conf.get("end_lr", lr))
    clip = conf["ppo_clip"]
    clip_scheduler = LinearScheduler(
        steps=steps,
        start=clip,
        end=conf.get("end_ppo_clip", clip))


    return ModelHolder(
            state_input,
            action_shape,
            name,
            train_planner,
            layers,
            weights,
            ent_reg,
            clip_scheduler,
            keras.callbacks.LearningRateScheduler(lr_scheduler.schedule),
            conf.get("expected_value", 0),
            conf.get("epochs", 1),
            conf.get("normalize_advantage", True),
            start_epoch)

def _gen_layers(lyrtype, state_input):
    if lyrtype == "conv":
        return _conv_layers(state_input)
    elif lyrtype == "conv2":
        return _conv_layers_dm(state_input)
    return _dense_layers(state_input)



class ModelHolder(object):
    _ACTION_OUT = "action_out"
    _VALUE_OUT = "value_out"

    def __init__(self, state_input, action_shape, name, train_planner, model_layers, weights, ent_reg, ppo_scheduler, lr_scheduler, expected_value=0, epochs = 1, normalize_adv=True, start_epoch=0):
        self.name = name
        self.train_planner = train_planner
        self.state_input = state_input
        self.n_actions=action_shape
        self.normalize_adv = normalize_adv
        self._action_out = layers.Dense(
                action_shape,
                activation="softmax",
                name=self._ACTION_OUT,
                activity_regularizer=ent_reg)(model_layers)
        self._value_out = layers.Dense(
                1,
                activation="linear",
                bias_initializer=keras.initializers.Constant(value=expected_value),
                name=self._VALUE_OUT)(model_layers)

        self.model = keras.Model(
                inputs = self.state_input,
                outputs = [self._action_out, self._value_out],
                name = "a2c_agent")
        self.model.summary()
        self.model.add_metric(ent_reg.last_loss, name="entropy_loss")
        self.ppo_scheduler = ppo_scheduler
        self.ppo_clip = tensorflow.Variable(ppo_scheduler.schedule(0))
        self.lr_scheduler = lr_scheduler
        optimizer = keras.optimizers.Adam(epsilon=1e-5)
        critic_loss = keras.losses.Huber()

        losses = {
                self._ACTION_OUT: self._ppo_loss,
                #self._ACTION_OUT: self._action_loss,
                self._VALUE_OUT: critic_loss}
        self.model.compile(
                optimizer=optimizer,
                loss=losses,
                loss_weights=weights)


        self._epochs = int(start_epoch / epochs) * epochs
        self._epochs_per_run = epochs
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

    def _action_cce_loss(self, y_true, y_pred):
        advantages = y_true[:, :1]
        actions_taken = y_true[:, 1:self.n_actions+1]
        return self._cce(actions_taken, y_pred, sample_weight=advantages)

    def _action_loss(self, y_true, y_pred):
        advantages = y_true[:, :1]
        actions_taken = y_true[:, 1:self.n_actions+1]
        action_prob = keras.backend.sum(y_pred * actions_taken, axis=1)
        return -tensorflow.reduce_mean(advantages * keras.backend.log(action_prob), axis=-1)


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
        if self._epochs % (10*self._epochs_per_run) == 0:
            log.info("saving model")
            path = "models/"+self.name+".h5"
            self.model.save(path)

    def load(self, model_name):
        self.model.load_weights(model_name)

    def train(self):
        memories = self.train_planner.release()
        if len(memories)  == 0:
            return
        states = numpy.vstack([m.state for m in memories])
        discounted_rewards = numpy.vstack([m.discounted_reward for m in memories])
        advantages = numpy.vstack([m.advantage for m in memories])
        if self.normalize_adv:
            #advantages = self._normalize(advantages)
            advantages = self._norm_inplace(advantages)
        actions_taken = numpy.vstack([m.action.onehot for m in memories])
        actions_preds = numpy.vstack([m.action.prediction for m in memories])
        self._train(states, discounted_rewards, advantages, actions_taken, actions_preds)

    def _norm_inplace(self, data):
        amp = numpy.sqrt(numpy.mean(numpy.square(data)))
        amp = max(amp, 1e-10)
        return data / amp

    def _normalize(self, data):
        mean = numpy.mean(data)
        stdev = numpy.std(data)
        stdev = max(stdev, 1e-10)
        return (data - mean) / stdev

    def _train(self, states, discounted_rewards, advantages, actions_taken, actions_pred):
        self.ppo_clip.assign(self.ppo_scheduler.schedule(self._epochs))
        log.info("lr scale "+str(1-self.ppo_scheduler.ratio_done(self._epochs)))
        action_y = numpy.hstack([advantages, actions_taken, actions_pred])
        batch_size = min(states.shape[0], self.train_planner.batch_size)
        y = {
                self._VALUE_OUT : discounted_rewards,
                self._ACTION_OUT : action_y }

        self.model.fit(
                x=states,
                y=y,
                epochs=self._epochs+self._epochs_per_run,
                batch_size=batch_size,
                shuffle=True,
                initial_epoch=self._epochs,
                callbacks=[self.tb_callback, self.lr_scheduler])
        self._epochs += self._epochs_per_run
        self._periodic_save()

    def apply(self, state):
        data = self.model.predict(state)
        return data[0][0], data[1][0]

class LinearScheduler(object):
    def __init__(self, steps, start=1.0, end=1.0):
        self.start = start
        self.end = end
        self.steps = float(steps)

    def schedule(self, epoch, *kargs, **kwargs):
        ratio = self.ratio_done(epoch)
        return (ratio * self.end) + ((1-ratio) * self.start)

    def ratio_done(self, epoch):
        return min(epoch/self.steps, 1.0)


class EntropyRegularizer(keras.regularizers.Regularizer):
    def __init__(self, strength, actions):
        super(EntropyRegularizer, self).__init__()
        self.strength = strength
        self._last_loss = 0
        self._base = math.log(actions)

    def __call__(self, x):
        #minus entropy. Since model seeks to minimize loss, this will
        #move probs to even values where entropy is highest
        losses = tensorflow.reduce_sum(x * tensorflow.math.log(x+1e-10), 1) / self._base
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

    def report_game(self, game_num, rewards, frames, total_rewards, avg_rewards):
        if self.writer is None:
            return
        with self.writer.as_default():
            tensorflow.summary.scalar('rewards_total', data=total_rewards, step=game_num)
            tensorflow.summary.scalar('rewards_last', data=rewards, step=game_num)
            tensorflow.summary.scalar('frames_total', data=frames, step=game_num)
            tensorflow.summary.scalar('rewards_avg', data=avg_rewards, step=game_num)
            self.writer.flush()

class OnPolicyPlanner(object):
    def __init__(self, batch_size=64, min_batches=4, max_batches=1e10):
        self.batch_size = batch_size
        self.min_batches = min_batches
        self.max_batches = max_batches
        self.memories = []

    def update(self, memories):
        self.memories.extend(memories)

    def release(self):
        batches = int(len(self.memories) / self.batch_size)
        if batches < self.min_batches:
            return []
        batches = min(batches, self.max_batches)
        to_release = random.sample(self.memories, batches * self.batch_size)
        self.memories = []
        return to_release


class TrainingPlanner(object):
    def __init__(self, max_mem=10000, min_mem=500, train_factor=2, batch_size=256):
        self.max_mem = max_mem
        self.min_mem = min_mem
        self.train_factor = train_factor
        self.batch_size = batch_size
        self.untrained = 0
        self.memories = []

    def update(self, memories):
        self.memories.extend(memories)
        if len(self.memories) - len(memories) < self.min_mem:
            return
        self.untrained += int(len(memories) * self.train_factor)
        to_discard = len(self.memories) - self.max_mem
        if to_discard > 0:
            self.memories = self.memories[to_discard:]

    def releasable(self):
        batches = int(self.untrained / self.batch_size)
        return min(batches*self.batch_size, len(self.memories))

    def release(self):
        to_train = self.releasable()
        if not to_train > 0:
            return []
        samples = random.sample(self.memories, to_train)
        self.untrained -= to_train
        return samples

class SampleSelector(object):
    def __init__(self, target_val, min_ratio=0.5):
        self.target_val = target_val
        self.min_ratio = min_ratio
        self._current_bias = 0

    def select(self, memories):
        accepted = [m for m in memories if m.hindsight_reward > self.min_ratio]
        remaining = [m for m in memories if m.hindsight_reward <= self.min_ratio]
        self._current_bias += len(accepted)
        to_add = min(len(remaining), self._current_bias)
        if to_add > 0:
            self._current_bias -= to_add
            accepted += random.sample(remaining, to_add)
        return accepted

def _dense_layers(state_input):
    x = layers.Flatten()(state_input)
    x = layers.Dense(512, activation="relu", name="hidden_1")(x)
    x = layers.Dense(256, activation="relu", name="hidden_2")(x)
    x = layers.Dense(64, activation="relu", name="hidden_3")(x)
    return x

def _conv_layers(state_input):
    x = state_input
    x = layers.Conv2D(64, 5, strides=2, activation="relu", name="conv_1", input_shape=x.shape[1:],padding="same")(x)
    x = layers.Conv2D(32, 3, strides=2, activation="relu", name="conv_2", padding="same")(x)
    x = layers.Conv2D(8, 3, activation="relu", name="conv_3", padding="same")(x)
    x = layers.Flatten()(x)
    return layers.Dense(32, activation="relu", name="dense_1")(x)

def _conv_layers_dm(state_input):
    x = state_input
    x = layers.Conv2D(16, 8, strides=3, activation="relu", name="conv_1", input_shape=x.shape[1:],padding="same")(x)
    x = layers.Conv2D(32, 4, strides=2, activation="relu", name="conv_2",padding="same")(x)
    x = layers.Flatten()(x)
    return layers.Dense(256, activation="relu", name="dense_2")(x)

