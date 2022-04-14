import logging
from collections import deque
import random

log = logging.getLogger(__name__)

class TestExecutor(object):
    def __init__(self, agent, game_callbacker):
        self.agent = agent
        self.game_callback = game_callbacker

    def run(self):
        while True:
            mems = self.agent.run()
            self.game_callback.update(mems)

class SequentialExecutor(object):
    def __init__(self, agents, model, train_planner, game_callbacker, samples, rerun_prob=0):
        self.agents = agents
        self.model = model
        self.train_planner = train_planner
        self.game_callback = game_callbacker
        self.samples = samples
        self.rerun_prob = rerun_prob


    def run(self):
        while True:
            self._run_agents()

    def _run_agents(self):
        for i, agent in enumerate(self.agents):
            self._run_agent(agent, i)
            if random.random() < self.rerun_prob:
                self._run_agent(agent, i)
        self.model.train()

    def _run_agent(self, agent, agent_num):
        mems = agent.run_frames(self.samples)
        if agent_num==0:
            self.game_callback.update(mems)
        else:
            self.game_callback.background_update(mems, agent_num)
        self.train_planner.update(mems)


class GameCallbacker(object):
    def __init__(self, plotter, reward_graph):
        self.memories = []
        self.plotter = plotter
        self.reward_grapher = reward_graph
        self.runs = 0
        self.frames = 0
        self.running_total_rewards = 0
        self.agent_run_rewards = {}
        self.game_averager = Averager(100)


    def background_update(self, memories, agent=0):
        self.frames += len(memories)
        rewards = 0
        for m in memories:
            rewards += m.unshaped_reward
            if m.done:
                self.runs += 1
                self.running_total_rewards += rewards
                rewards += self.agent_run_rewards.get(agent, 0)
                self.game_averager.add(rewards)
                self.agent_run_rewards[agent] = 0
                rewards = 0
        self.running_total_rewards += rewards
        self.agent_run_rewards[agent] = rewards + self.agent_run_rewards.get(agent, 0)

        if self.plotter is not None:
            self.plotter.refresh()


    def update(self, memories):
        self.background_update(memories)
        last_done = 0
        for i, mem in enumerate(memories):
            if mem.done:
                self.memories.extend(memories[last_done:i+1])
                last_done = i+1
                self._callback()
                self.memories = []
        self.memories.extend(memories[last_done:])

    def _callback(self):
        tot_reward = sum(m.unshaped_reward for m in self.memories)
        if self.plotter is not None:
            self.plotter.plot_mems(self.memories)
        if self.reward_grapher is not None:
            self.reward_grapher.report_game(self.runs, tot_reward, self.frames, self.running_total_rewards, self.game_averager.average())
        log.info("Run %s frames %s score %s", self.runs, len(self.memories), tot_reward)

class Averager(object):
    def __init__(self, n=100):
        self._data = deque(maxlen=n)
        self._cur = 0.0

    def add(self, val):
        if len(self._data) >= self._data.maxlen:
            self._cur -= self._data.popleft()
        self._data.append(val)
        self._cur += val

    def average(self):
        return self._cur / len(self._data)
