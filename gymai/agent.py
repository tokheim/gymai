import numpy
import random
import logging
import matplotlib.pyplot as plt
from collections import deque
import cv2
log = logging.getLogger(__name__)

class A2CAgent:
    def __init__(self, env, model, state_converter, discount_gamma=0.96, end_reward=-1, discount_mix=0.9, render=True):
        self.env = env
        self.model = model
        self.state_converter = state_converter
        self.action_space = self.env.action_space.n
        self.runs = 0
        self.end_reward = end_reward
        self.discount_gamma = discount_gamma
        self.discount_mix = discount_mix
        self._should_render = render

    def generate_predictions(self, state):
        action_pred, val = self.model.apply(state)
        action = Action(action_pred)
        memory = Memory(state, action, val)
        return memory

    def run(self):
        self.state_converter.reset()
        state = self.state_converter.convert(self.env.reset())
        self.env.seed(random.randint(0, 1e5))
        state, _, _, _  = self.env.step(1)#breakout force fire btn
        state = self.state_converter.convert(state)
        done = False
        memories = []
        tot_reward = 0
        while not done:
            mem = self.generate_predictions(state)
            self._render(mem)
            state, reward, done, info = self.env.step(mem.action.n)
            state = self.state_converter.convert(state)
            mem.reward = min(reward, 1.)
            memories.append(mem)
            tot_reward += reward
            if self._breakout_end(info):
                break
        self._tag_mems(memories)
        self.runs += 1
        self.model.reward_callback.report_game(self.runs, tot_reward)
        log.info("Run %s frames %s score %s", self.runs, len(memories), tot_reward)
        return memories

    def _render(self, memory):
        if self._should_render:
            self.env.render()

    def _plot(self, vals, actions, rewards, advantages):
        if self._should_render:
            plt.cla()
            plt.axis([0, len(vals), self.end_reward, 3])
            x = numpy.arange(0, len(vals))
            actions_ent = [a.entropy() for a in actions]
            actions_prob = [a.action_prob for a in actions]
            plt.plot(x, advantages, c='y')
            plt.plot(x, actions_ent, c='b')
            plt.plot(x, actions_prob, c='k')
            plt.plot(x, rewards, c='g')
            plt.plot(x, vals, c='r')
            plt.pause(0.0001)

    def _breakout_end(self, info):
        return info.get('ale.lives') == 4

    def _tag_mems(self, memories):
        memories[-1].reward += self.end_reward
        next_vals = [m.value_prediction for m in memories[1:]] + [0]

        next_val_pred = numpy.vstack(next_vals)
        val_pred = numpy.vstack([m.value_prediction for m in memories])
        rewards = numpy.vstack([m.reward for m in memories])

        discounted_rewards = (self.discount_gamma * next_val_pred) + rewards
        discounted_rewards = self.discount_mix*discounted_rewards
        real_discount = self._discount_real(rewards)
        discounted_rewards += (1-self.discount_mix) * real_discount

        advantages = discounted_rewards - val_pred
        for memory, discounted_reward, advantage in zip(memories, discounted_rewards, advantages):
            memory.discounted_reward = discounted_reward
            memory.advantage = advantage



        self._plot(val_pred, [m.action for m in memories], real_discount, advantages)

    def _discount_real(self, rewards):
        current_reward = 0
        discounted = numpy.zeros(rewards.shape, dtype="float")
        for i, reward in enumerate(reversed(rewards)):
            current_reward = reward + (self.discount_gamma * current_reward)
            discounted[-i-1] = current_reward
        return discounted


    def train(self, memories):
        if len(memories) == 0:
            return
        states = numpy.vstack([m.state for m in memories])
        discounted_rewards = numpy.vstack([m.discounted_reward for m in memories])
        advantages = numpy.vstack([m.advantage for m in memories])
        actions_taken = numpy.vstack([m.action.onehot for m in memories])
        actions_preds = numpy.vstack([m.action.prediction for m in memories])
        self.model.train(states, discounted_rewards, advantages, actions_taken, actions_preds)



class Action(object):
    def __init__(self, prediction):
        self.prediction = prediction
        self._action = numpy.random.choice(len(prediction), p=prediction)

    @property
    def n(self):
        return self._action

    @property
    def action_prob(self):
        return self.prediction[self._action]

    @property
    def onehot(self):
        vector = numpy.zeros(self.prediction.shape)
        vector[self._action] = 1
        return vector

    def entropy(self):
        return - sum(self.prediction * numpy.log(numpy.clip(self.prediction, 1e-10, None)))

class Memory(object):
    def __init__(self, state, action, value_prediction, reward=None, discounted_reward=None, advantage=None):
        self.state = state
        self.action = action
        self.reward = reward
        self.value_prediction = value_prediction
        self.discounted_reward = discounted_reward
        self.advantage = advantage

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

    def save_img(self, img):
        cv2.imwrite("images/test.jpg", img)


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
        self.untrained += len(memories) * int(self.train_factor)
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
