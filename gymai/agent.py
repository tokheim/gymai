import numpy
import logging
import matplotlib.pyplot as plt
from collections import deque
import random
import cv2
log = logging.getLogger(__name__)

class A2CAgent:
    def __init__(self, env, model, state_converter, game_hacks, discount_gamma=0.96, end_reward=-1, gae_lambda=0.95, max_reward=1, render=True, plotter=None, max_action=False):
        self.env = env
        self.model = model
        self.game_hacks = game_hacks
        self.state_converter = state_converter
        self.action_space = self.env.action_space.n
        self.max_reward = max_reward
        self.runs = 0
        self.end_reward = end_reward
        self.discount_gamma = discount_gamma
        self._should_render = render
        self._plotter = plotter
        self.gae_lambda = gae_lambda
        self._max_action = max_action

    def generate_predictions(self, state):
        action_pred, val = self.model.apply(state)
        action = Action(action_pred)
        memory = Memory(state, action, val)
        if self._max_action:
            action.max_action()
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
            tot_reward += reward
            state = self.state_converter.convert(state)
            reward += self.game_hacks.custom_reward(info)
            mem.reward = min(reward, self.max_reward)
            memories.append(mem)
            if self.game_hacks.should_end(info):
                break
        memories[-1].reward += self.end_reward
        self.game_hacks.on_end(memories)
        self._tag_mems(memories)
        self.runs += 1
        self.model.reward_callback.report_game(self.runs, tot_reward, len(memories))
        log.info("Run %s frames %s score %s", self.runs, len(memories), tot_reward)
        return memories

    def _render(self, memory):
        if self._should_render:
            self.env.render()

    def _plot(self, vals, actions, rewards, advantages):
        if self._plotter:
            self._plotter.plot(vals, actions, rewards, advantages)

    def _tag_mems(self, memories):
        val_pred = numpy.vstack([m.value_prediction for m in memories])
        rewards = numpy.vstack([m.reward for m in memories])

        advantages = self._gen_adv_est(rewards, val_pred)
        real_discount = self._discount_real(rewards)
        discounted_rewards = self._discounted_target(val_pred, rewards)
        discounted_rewrads = advantages + val_pred

        for memory, discounted_reward, advantage, real_reward in zip(memories, discounted_rewards, advantages, real_discount):
            memory.discounted_reward = discounted_reward
            memory.advantage = advantage
            memory.hindsight_reward = real_reward



        self._plot(val_pred, [m.action for m in memories], real_discount, advantages)

    def _discounted_target(self, val_pred, rewards):
        next_vals = numpy.roll(val_pred, -1)
        next_vals[-1] = 0
        return (self.discount_gamma * next_vals) + rewards

    def _gen_adv_est(self, rewards, val_pred):
        advantages = numpy.zeros(rewards.shape)
        gae = 0
        last_val = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + (self.discount_gamma * last_val) - val_pred[i]
            last_val = val_pred[i]
            gae = delta + (self.discount_gamma * self.gae_lambda * gae)
            advantages[i] = gae# + val_pred[i]
        return advantages


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
        self.random_action()

    @property
    def n(self):
        return self._action

    def max_action(self):
        self._action = numpy.argmax(self.prediction)

    def random_action(self):
        self._action = numpy.random.choice(len(self.prediction), p=self.prediction)

    @property
    def action_prob(self):
        return self.prediction[self._action]

    @property
    def onehot(self):
        vector = numpy.zeros(self.prediction.shape)
        vector[self._action] = 1
        return vector

    def entropy(self):
        ents = self.prediction * numpy.log(numpy.clip(self.prediction, 1e-10, None))
        return - sum(ents) / numpy.log(len(self.prediction))

class Memory(object):
    def __init__(self, state, action, value_prediction, reward=None, discounted_reward=None, advantage=None, hindsight_reward=None):
        self.state = state
        self.action = action
        self.reward = reward
        self.value_prediction = value_prediction
        self.discounted_reward = discounted_reward
        self.advantage = advantage
        self.hindsight_reward = hindsight_reward

class ReshapeConverter(object):
    def __init__(self, original_shape, history_picks):
        self.shape = tuple([1, len(history_picks)] + list(original_shape))
        self._state=numpy.zeros(tuple([1, max(history_picks)+1] + list(original_shape)))
        self._hists=history_picks

    def convert(self, state):
        numpy.roll(self._state, 1, axis=1)
        self._state[0,0,:] = state
        return self._state[:,self._hists,:]

    def reset(self):
        self._state = numpy.zeros(self._state.shape)

class ImageConverter(object):
    def __init__(self, dim, history_picks, name="test"):
        self.name = name
        self._img_dim = dim
        self._state = numpy.zeros(tuple([1]+list(self._img_dim)+[max(history_picks)+1]))
        self._sample_rate = 0.001
        self._sampled = 0
        self._max_sample = 20
        self._bw_trans = [0.299, 0.587, 0.114]
        self.shape = tuple([1]+list(self._img_dim)+[len(history_picks)])
        self._history_picks = history_picks
        self.reset()

    @property
    def history(self):
        return self.shape[-1]

    def reset(self):
        self._state = numpy.zeros(self._state.shape, dtype="float")

    def convert(self, state):
        img = cv2.resize(state, dsize=self._img_dim[::-1], interpolation=cv2.INTER_CUBIC)
        bw = self._bw_trans[0]*img[:,:,0] + self._bw_trans[1]*img[:,:,1] + self._bw_trans[2]*img[:,:,2]
        scaled = bw / 255.0
        self._state = numpy.roll(self._state, 1, axis=-1)
        self._state[:,:,:,0] = scaled
        if random.random() < self._sample_rate:
            self.save_img()
        return self._state[:,:,:,self._history_picks]

    def save_img(self):
        bw = numpy.vstack([self._state[0,:,:,i] for i in range(self.history)]) * 255.0
        img = numpy.zeros((bw.shape[0], bw.shape[1], 3))
        for i in range(3):
            img[:,:,i] = bw
        path = "images/"+self.name+"-"+str(self._sampled%self._max_sample)+".png"
        cv2.imwrite(path, img)
        self._sampled += 1

def create_converter(conf, env_shape, name):
    history = conf.get("history", 1)
    skips = conf.get("skip", 0)
    hist_picks = list(range(0, history*(skips+1), skips+1))
    if conf.get("type") == "flat":
        return ReshapeConverter(env_shape, hist_picks)
    return ImageConverter((conf["height"], conf["width"]), hist_picks, name=name)

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
        self._setup(self.ax[1], xlim, [-0.05, 1.05])
        self.ax[1].plot(x, actions_ent, c='b')
        self.ax[1].plot(x, actions_prob, c='k')
        ylim = self._axis([advantages], -1, 1, buff=0.1)
        self._setup(self.ax[2], xlim, ylim)
        self.ax[2].plot(x, advantages)

        plt.pause(0.000001)
