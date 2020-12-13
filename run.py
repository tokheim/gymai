import argparse
import logging
import gym
from gymai.agent import A2CAgent, ReshapeConverter, ImageConverter, TrainingPlanner
from gymai.model import ModelHolder

def main(args):
    env = gym.make(args.game)
    train_planner = TrainingPlanner(batch_size=256, train_factor=8)
    #converter = ReshapeConverter(env.observation_space.shape)
    converter = ImageConverter((84, 64), history=2)
    model = ModelHolder(converter.shape[1:], env.action_space.n, batch_size=train_planner.batch_size)
    agent = A2CAgent(env, model, converter)
    while True:
        mems = agent.run()
        train_planner.update(mems)
        agent.train(train_planner.release())


if __name__ == '__main__':
    parser = argparse.ArgumentParser('AI gymnastics')
    parser.add_argument('-g', '--game', default='CartPole-v1', type=str)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    main(args)
