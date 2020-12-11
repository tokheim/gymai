import numpy
import logging
from collections import deque
import cv2
log = logging.getLogger(__name__)

class A2CAgent:
    def __init__(self, env, model, state_converter, discount_gamma=0.9):
        self.env = env
        self.model = model
        self.state_converter = state_converter
        self.action_space = self.env.action_space.n
        self.runs = 0
        self.discount_gamma = discount_gamma

    def generate_predictions(self, state):
        action_pred, val = self.model.apply(state)
        action = Action(action_pred)
        memory = Memory(state, action, val)
        return memory

    def run(self):
        self.state_converter.reset()
        state = self.state_converter.convert(self.env.reset())
        done = False
        memories = []
        tot_reward = 0
        while not done:
            mem = self.generate_predictions(state)
            state, reward, done, _ = self.env.step(mem.action.n)
            state = self.state_converter.convert(state)
            mem.reward = reward
            memories.append(mem)
            tot_reward += reward
        self.runs += 1
        self.model.reward_callback.add_rewards(tot_reward)
        log.info("Run %s frames %s score %s", self.runs, len(memories), tot_reward)
        return memories

    def _train_data(self, memories):
        next_val_pred = numpy.vstack([m.value_prediction for m in memories[1:]])
        memories = memories[:-1]
        states = numpy.vstack([m.state for m in memories])
        actions_taken = numpy.vstack([m.action.onehot for m in memories])
        actions_pred = numpy.vstack([m.action.prediction for m in memories])
        val_pred = numpy.vstack([m.value_prediction for m in memories])
        rewards = numpy.vstack([m.reward for m in memories])

        discounted_rewards = (self.discount_gamma * next_val_pred) + rewards
        discounted_rewards[-1] = -50
        advantages = discounted_rewards - val_pred

        return states, discounted_rewards, advantages, actions_taken

    def train(self, memories):
        states, discounted_rewards, advantages, actions_taken = self._train_data(memories)
        self.model.train(states, discounted_rewards, advantages, actions_taken)



class Action(object):
    def __init__(self, prediction):
        self.prediction = prediction
        self._action = numpy.random.choice(len(prediction), p=prediction)

    @property
    def n(self):
        return self._action

    @property
    def onehot(self):
        vector = numpy.zeros(self.prediction.shape)
        vector[self._action] = 1
        return vector

class Memory(object):
    def __init__(self, state, action, value_prediction, reward=None):
        self.state = state
        self.action = action
        self.reward = reward
        self.value_prediction = value_prediction

class ReshapeConverter(object):
    def __init__(self, original_shape):
        self.shape = tuple([1] + list(original_shape))

    def convert(self, state):
        return numpy.reshape(state, self.shape)

    def reset(self):
        pass

class ImageConverter(object):
    def __init__(self, dim, history=1):
        self._img_dim = dim
        self._state = numpy.zeros(tuple([1]+list(self._img_dim)+[history]))
        self.reset()

    @property
    def shape(self):
        return self._state.shape

    @property
    def history(self):
        return self.shape[-1]

    def reset(self):
        self._state = numpy.zeros(self.shape, dtype="float")

    def convert(self, state):
        img = cv2.resize(state, dsize=self._img_dim[::-1])
        bw = 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]
        scaled = bw / 255.0
        self._state = numpy.roll(self._state, 1, axis=-1)
        self._state[:,:,:,0] = scaled
        return self._state
