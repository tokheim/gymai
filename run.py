import argparse
import logging
import datetime
import gym
import yaml
from gymai.agent import create_converter, Plotter, AgentBuilder, RewardShaper
from gymai.model import create_model, TrainingPlanner, SampleSelector, OnPolicyPlanner
from gymai import executor
import logging


def main(args):
    conf = _load_conf(args.config)
    _load_extra_envs(conf)
    env = gym.make(conf["env"])
    train_planner = OnPolicyPlanner(**conf["planner"])
    converter = create_converter(conf["converter"], env.observation_space.shape, args.name)
    model = create_model(conf["model"], converter.shape[1:], env.action_space.n, train_planner, args.name)
    env.close()
    reward_shaper = RewardShaper(**conf["rewardShaper"])
    plotter = None
    if args.plot:
        plotter = Plotter(args.name, reward_shaper)
    if args.load_model is not None:
        model.load(args.load_model)
    agent_builder = AgentBuilder(conf["env"], model, converter, reward_shaper, conf["agent"])
    game_callbacker = executor.GameCallbacker(plotter, model.reward_callback)
    runner = None
    if args.test:
        runner = executor.TestExecutor(agent_builder.build(render=args.render), game_callbacker)
    else:
        workers = conf["executor"].get("workers", 1)
        sample_size = conf["executor"]["samples"]
        agents = agent_builder.build_agents(conf["executor"].get("workers", 1), render=args.render)
        runner = executor.SequentialExecutor(agents, model, train_planner, game_callbacker, sample_size)

    runner.run()

def _load_conf(filename):
    with open(filename, 'r') as f:
        return yaml.load(f)

def _load_extra_envs(conf):
    if not conf.get("extended_envs", False):
        return
    try:
        from pytetris import tetrisgym
        import vizdoomgym
    except ImportError as e:
        log.warn("unable to load pytetris")


if __name__ == '__main__':
    default_name = "run_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    parser = argparse.ArgumentParser('AI gymnastics')
    parser.add_argument('-c', '--config', default='CartPole-v1', type=str)
    parser.add_argument('-n', '--name', default=default_name, type=str)
    parser.add_argument('-l', '--load-model', default=None, type=str)
    parser.add_argument('-r', '--render', action="store_true")
    parser.add_argument('-p', '--plot', action="store_true")
    parser.add_argument('-t', '--test', action="store_true")
    parser.add_argument('-m', '--max-action', action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    main(args)
