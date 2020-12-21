import numpy
import logging
import matplotlib.pyplot as plt
from collections import deque
import random
import cv2
log = logging.getLogger(__name__)

class A2CAgent:
    def __init__(self, env, model, state_converter, game_hacks, discount_gamma=0.96, end_reward=-1, discount_mix=0.9, max_reward=1, render=True, plotter=None):
        self.env = env
        self.model = model
        self.game_hacks = game_hacks
        self.state_converter = state_converter
        self.action_space = self.env.action_space.n
        self.max_reward = max_reward
        self.runs = 0
        self.end_reward = end_reward
        self.discount_gamma = discount_gamma
        self.discount_mix = discount_mix
        self._should_render = render
        self._plotter = plotter

    def generate_predictions(self, state):
        action_pred, val = self.model.apply(state)
        action = Action(action_pred)
        memory = Memory(state, action, val)
        return memory

    def run(self):
        self.state_converter.reset()
        state = self.env.reset()
        state = self.game_hacks.on_start(self.env, state)
        state = self.state_converter.convert(state)
        done = False
        memories = []
        tot_reward = 0
        while not done:
            mem = self.generate_predictions(state)
            self._render(mem)
            state, reward, done, info = self.env.step(mem.action.n)
            state = self.state_converter.convert(state)
            mem.reward = min(reward, self.max_reward)
            memories.append(mem)
            tot_reward += reward
            if self.game_hacks.should_end(info):
                break
        memories[-1].reward += self.end_reward
        self.game_hacks.on_end(memories)
        self._tag_mems(memories)
        self.runs += 1
        self.model.reward_callback.report_game(self.runs, tot_reward)
        log.info("Run %s frames %s score %s", self.runs, len(memories), tot_reward)
        return memories

    def _render(self, memory):
        if self._should_render:
            self.env.render()

    def _plot(self, vals, actions, rewards, advantages):
        if self._plotter:
            self._plotter.plot(vals, actions, rewards, advantages)

    def _tag_mems(self, memories):
        next_vals = [m.value_prediction for m in memories[1:]] + [0]

        next_val_pred = numpy.vstack(next_vals)
        val_pred = numpy.vstack([m.value_prediction for m in memories])
        rewards = numpy.vstack([m.reward for m in memories])

        discounted_rewards = (self.discount_gamma * next_val_pred) + rewards
        #predicted_future = (val_pred-rewards) / self.discount_gamma
        #advantages = next_val_pred - predicted_future

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
    def __init__(self, dim, history=1, name="test"):
        self.name = name
        self._img_dim = dim
        self._state = numpy.zeros(tuple([1]+list(self._img_dim)+[history]))
        self._sample_rate = 0.001
        self._sampled = 0
        self._max_sample = 20
        self._bw_trans = [0.299, 0.587, 0.114]
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
        img = cv2.resize(state, dsize=self._img_dim[::-1], interpolation=cv2.INTER_CUBIC)
        bw = self._bw_trans[0]*img[:,:,0] + self._bw_trans[1]*img[:,:,1] + self._bw_trans[2]*img[:,:,2]
        scaled = bw / 255.0
        self._state = numpy.roll(self._state, 1, axis=-1)
        self._state[:,:,:,0] = scaled
        if random.random() < self._sample_rate:
            self.save_img()
        return self._state

    def save_img(self):
        bw = numpy.vstack([self._state[0,:,:,i] for i in range(self.history)]) * 255.0
        img = numpy.zeros((bw.shape[0], bw.shape[1], 3))
        for i in range(3):
            img[:,:,i] = bw
        path = "images/"+self.name+"-"+str(self._sampled%self._max_sample)+".png"
        cv2.imwrite(path, img)
        self._sampled += 1

class Plotter(object):
    def __init__(self, name):
        ax = []
        fig, ax = plt.subplots(3, 1)
        fig.canvas.set_window_title(name)
        self.ax = ax

    def _setup(self, ax, xlim, ylim):
        ax.cla()
        ax.set(xlim=xlim, ylim=ylim)
        ax.grid()

    def _axis(self, arrays, minval=1e20, maxval=-1e20, buff=0):
        maxes = [max(arr) for arr in arrays]
        mines = [min(arr) for arr in arrays]
        buff_val = max([buff*(a-b) for a, b in zip(maxes, mines)])
        return [min(minval, *mines) - buff_val, max(maxval, *maxes) + buff_val]

    def plot(self, vals, actions, rewards, advantages):
        xlim = [0, len(vals)]
        ylim = self._axis([rewards, vals], buff=0.1)
        self._setup(self.ax[0], xlim, ylim)
        x = numpy.arange(0, len(vals))
        self.ax[0].plot(x, rewards, c='g')
        self.ax[0].plot(x, vals, c='r')

        actions_ent = [a.entropy() for a in actions]
        actions_prob = [a.action_prob for a in actions]
        self._setup(self.ax[1], xlim, [0, 2])
        self.ax[1].plot(x, actions_ent, c='b')
        self.ax[1].plot(x, actions_prob, c='k')
        ylim = self._axis([advantages], -1, 1, buff=0.1)
        self._setup(self.ax[2], xlim, ylim)
        self.ax[2].plot(x, advantages)



        plt.pause(0.0001)

def create_converter(conf, env_shape, name):
    if conf.get("type") == "flat":
        return ReshapeConverter(env_shape)
    return ImageConverter((conf["height"], conf["width"]), conf["history"], name=name)
