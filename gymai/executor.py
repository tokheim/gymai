import logging

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
    def __init__(self, agents, model, train_planner, game_callbacker, samples):
        self.agents = agents
        self.model = model
        self.train_planner = train_planner
        self.game_callback = game_callbacker
        self.samples = samples


    def run(self):
        while True:
            self._run_agents()

    def _run_agents(self):
        for i, agent in enumerate(self.agents):
            self._run_agent(agent, i==0)
        self.model.train()

    def _run_agent(self, agent, should_callback):
        mems = agent.run_frames(self.samples)
        if should_callback:
            self.game_callback.update(mems)
        self.train_planner.update(mems)


class GameCallbacker(object):
    def __init__(self, plotter, reward_graph):
        self.memories = []
        self.plotter = plotter
        self.reward_grapher = reward_graph
        self.runs = 0

    def update(self, memories):
        last_done = 0
        for i, mem in enumerate(memories):
            if mem.done:
                self.memories.extend(memories[last_done:i+1])
                last_done = i+1
                self._callback()
                self.memories = []
        self.memories.extend(memories[last_done:])

    def _callback(self):
        self.runs += 1
        tot_reward = sum(m.unshaped_reward for m in self.memories)
        if self.plotter is not None:
            self.plotter.plot_mems(self.memories)
        if self.reward_grapher is not None:
            self.reward_grapher.report_game(self.runs, tot_reward, len(self.memories))
        log.info("Run %s frames %s score %s", self.runs, len(self.memories), tot_reward)

