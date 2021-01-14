import numpy
import logging
import matplotlib.pyplot as plt
from collections import deque
import random
import cv2
import gym
log = logging.getLogger(__name__)

class AgentBuilder(object):
    def __init__(self, envname, model, state_conv, reward_shaper, agent_conf):
        self.envname = envname
        self.model = model
        self.state_conv = state_conv
        self.agent_conf = agent_conf
        self.reward_shaper = reward_shaper

    def build(self, render=False):
        converter = self.state_conv.copy()
        env = gym.make(self.envname)
        return A2CAgent(env, self.model, converter, self.reward_shaper, render=render, **self.agent_conf)

    def build_agents(self, n, render=False):
        agents = []
        for i in range(n):
            agent = self.build(render and i == 0)
            agents.append(agent)
        return agents


class A2CAgent:
    def __init__(self, env, model, state_converter, reward_shaper, end_reward=-1, max_reward=1, render=True, max_action=False, stickiness = 0):
        self.env = env
        self.model = model
        self.state_converter = state_converter
        self.action_space = self.env.action_space.n
        self.max_reward = max_reward
        self.reward_shaper = reward_shaper
        self.runs = 0
        self.end_reward = end_reward
        self._should_render = render
        self._max_action = max_action
        self._action_repeat = stickiness+1
        self._last_state = self.state_converter.convert(env.reset())

    def generate_predictions(self, state):
        action_pred, val = self.model.apply(state)
        action = Action(action_pred)
        memory = Memory(state, action, val)
        if self._max_action:
            action.max_action()
        return memory

    def reset(self):
        self.state_converter.reset()
        self._last_state = self.state_converter.convert(self.env.reset())

    def run_frames(self, frames):
        memories = []
        remaining = frames
        while remaining > 0:
            memories.extend(self.run(remaining))
            remaining = frames - len(memories)
        return memories

    def run(self, max_frames=1e6):
        self.state_converter.reset()
        done = False
        memories = []
        state = self._last_state
        while not done:
            mem = self.generate_predictions(state)
            self._render(mem)
            state, reward, done, info = self._perform_action(mem.action)
            state = self.state_converter.convert(state)
            mem.unshaped_reward = reward
            mem.reward = min(reward, self.max_reward)
            mem.done = done
            memories.append(mem)
            if len(memories) > max_frames:
                break
        end_val = 0
        if done:
            memories[-1].reward += self.end_reward
            self.reset()
        else:
            self._last_state = state
            end_val = self.generate_predictions(state).value_prediction

        self.reward_shaper.tag_mems(memories, end_value=end_val)
        return memories

    def _perform_action(self, action):
        cum_reward = 0
        state = None
        done = False
        info = {}
        for _ in range(self._action_repeat):
            state, reward, done, info = self.env.step(action.n)
            cum_reward += reward
            if done:
                break
        return state, cum_reward, done, info

    def _render(self, memory):
        if self._should_render:
            self.env.render()

class RewardShaper(object):
    def __init__(self, discount_gamma, gae_lambda):
        self.discount_gamma = discount_gamma
        self.gae_lambda = gae_lambda

    def tag_mems(self, memories, end_value=0):
        val_pred = numpy.vstack([m.value_prediction for m in memories])
        rewards = numpy.vstack([m.reward for m in memories])

        advantages = self._gen_adv_est(rewards, val_pred, end_value)
        #discounted_rewards = self._discounted_target(val_pred, rewards)
        discounted_rewards = advantages + val_pred

        for memory, discounted_reward, advantage in zip(memories, discounted_rewards, advantages):
            memory.discounted_reward = discounted_reward
            memory.advantage = advantage

    def real_discount(self, memories):
        rewards = numpy.vstack([m.reward for m in memories])
        return self._discount_real(rewards)

    def _discounted_target(self, val_pred, rewards):
        next_vals = numpy.roll(val_pred, -1)
        next_vals[-1] = 0
        return (self.discount_gamma * next_vals) + rewards

    def _gen_adv_est(self, rewards, val_pred, end_val):
        advantages = numpy.zeros(rewards.shape)
        gae = 0
        last_val = end_val
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + (self.discount_gamma * last_val) - val_pred[i]
            last_val = val_pred[i]
            gae = delta + (self.discount_gamma * self.gae_lambda * gae)
            advantages[i] = gae
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
    def __init__(self, state, action, value_prediction, reward=None, discounted_reward=None, advantage=None, done=False):
        self.state = state
        self.action = action
        self.reward = reward
        self.value_prediction = value_prediction
        self.discounted_reward = discounted_reward
        self.advantage = advantage
        self.done = done
        self.unshaped_reward = reward

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

    def copy(self):
        return ReshapeConverter(self.shape[2:], self._hists)

class ImageConverter(object):
    def __init__(self, dim, history_picks, name="test", colored=False):
        self.name = name
        self.colored = colored
        self._history_picks = history_picks
        self._img_dim = dim
        self._state = numpy.zeros(tuple([1]+list(self._img_dim)+[max(self._history_picks)+1]))
        self._sample_rate = 0.001
        self._sampled = 0
        self._max_sample = 20
        self._bw_trans = [0.299, 0.587, 0.114]
        self.shape = tuple([1]+list(self._img_dim)+[len(self._history_picks)])
        self.reset()

    def copy(self):
        return ImageConverter(self._img_dim, self._history_picks, self.name, self.colored)

    @property
    def history(self):
        return self.shape[-1]

    def reset(self):
        self._state = numpy.zeros(self._state.shape, dtype="float")

    def convert(self, state):
        img = cv2.resize(state, dsize=self._img_dim[::-1], interpolation=cv2.INTER_CUBIC)
        if not self.colored:
            img = self._blackwhite(img,0) + self._blackwhite(img, 1) + self._blackwhite(img, 2)
        scaled = img / 255.0
        self._state = numpy.roll(self._state, self.color_bands, axis=-1)
        self._state[:,:,:,0:self.color_bands] = scaled
        if random.random() < self._sample_rate:
            self.save_img()
        return self._state[:,:,:,self._history_picks]

    def _blackwhite(self, img, band):
        return self._bw_trans[band]*img[:,:,band:band+1]

    @property
    def color_bands(self):
        if self.colored:
            return 3
        return 1

    @staticmethod
    def shape_hist_picks(history_picks, colored):
        if not colored:
            return history_picks
        picks = []
        for hist in history_picks:
            offset = hist*3
            picks.extend([offset, offset+1, offset+2])
        return picks


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
    colored = conf.get("colored", False)
    hist_picks = ImageConverter.shape_hist_picks(hist_picks, colored)
    return ImageConverter((conf["height"], conf["width"]), hist_picks, colored=colored, name=name)

class Plotter(object):
    def __init__(self, name, reward_shaper):
        ax = []
        fig, ax = plt.subplots(3, 1)
        fig.canvas.set_window_title(name)
        self.ax = ax
        self.reward_shaper = reward_shaper

    def _setup(self, ax, xlim, ylim):
        ax.cla()
        ax.set(xlim=xlim, ylim=ylim)
        ax.grid()

    def _axis(self, arrays, minval=1e20, maxval=-1e20, buff=0):
        maxes = [max(arr) for arr in arrays]
        mines = [min(arr) for arr in arrays]
        buff_val = max([buff*(a-b) for a, b in zip(maxes, mines)])
        return [min(minval, *mines) - buff_val, max(maxval, *maxes) + buff_val]

    def plot_mems(self, memories):
        vals = numpy.vstack([m.value_prediction for m in memories])
        actions = [m.action for m in memories]
        rewards = self.reward_shaper.real_discount(memories)
        advantages = numpy.vstack([m.advantage for m in memories])
        self.plot(vals, actions, rewards, advantages)

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
