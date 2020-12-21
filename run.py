import argparse
import logging
import datetime
import gym
import yaml
from gymai.agent import A2CAgent, create_converter, Plotter
from gymai.model import create_model, TrainingPlanner
from gymai import gamehacks

def main(args):
    conf = _load_conf(args.config)
    env = gym.make(conf["env"])
    train_planner = TrainingPlanner(**conf["planner"])
    converter = create_converter(conf["converter"], env.observation_space.shape, args.name)
    model = create_model(conf["model"], converter.shape[1:], env.action_space.n, train_planner, args.name)
    plotter = None
    if args.plot:
        plotter = Plotter(args.name)
    if args.load_model is not None:
        model.load(args.load_model)
    hacks = gamehacks.NoopGameHacks()
    if conf.get("gamehacks", False):
        hacks = gamehacks.load_hacks(conf["env"])
    agent = A2CAgent(env, model, converter, hacks, render=args.render, plotter=plotter, **conf["agent"])
    while True:
        mems = agent.run()
        train_planner.update(mems)
        model.train()


def _load_conf(filename):
    with open(filename, 'r') as f:
        return yaml.load(f)

if __name__ == '__main__':
    default_name = "run_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    parser = argparse.ArgumentParser('AI gymnastics')
    parser.add_argument('-c', '--config', default='CartPole-v1', type=str)
    parser.add_argument('-n', '--name', default=default_name, type=str)
    parser.add_argument('-l', '--load-model', default=None, type=str)
    parser.add_argument('-r', '--render', action="store_true")
    parser.add_argument('-p', '--plot', action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    main(args)
