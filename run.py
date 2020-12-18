import argparse
import logging
import datetime
import gym
from gymai.agent import A2CAgent, ReshapeConverter, ImageConverter, TrainingPlanner
from gymai.model import ModelHolder

def main(args):
    env = gym.make(args.game)
    train_planner = TrainingPlanner(batch_size=512, train_factor=4, max_mem=50000)
    #converter = ReshapeConverter(env.observation_space.shape)
    converter = ImageConverter((84, 64), history=2)
    model = ModelHolder(converter.shape[1:], env.action_space.n, batch_size=train_planner.batch_size, name=args.name)
    if args.load_model is not None:
        model.load(args.load_model)
    agent = A2CAgent(env, model, converter)
    while True:
        mems = agent.run()
        train_planner.update(mems)
        agent.train(train_planner.release())


if __name__ == '__main__':
    default_name = "run_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    parser = argparse.ArgumentParser('AI gymnastics')
    parser.add_argument('-g', '--game', default='CartPole-v1', type=str)
    parser.add_argument('-n', '--name', default=default_name, type=str)
    parser.add_argument('-l', '--load-model', default=None, type=str)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    main(args)
