import argparse
import logging
import gym
from gymai.agent import A2CAgent, ReshapeConverter, ImageConverter
from gymai.model import ModelHolder

def main(args):
    env = gym.make(args.game)
    #converter = ReshapeConverter(env.observation_space.shape)
    converter = ImageConverter((42, 32), history=2)
    print(converter.shape)
    model = ModelHolder(converter.shape[1:], env.action_space.n)
    agent = A2CAgent(env, model, converter)
    for _ in range(1000):
        mems = agent.run()
        agent.train(mems)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('AI gymnastics')
    parser.add_argument('-g', '--game', default='CartPole-v1', type=str)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    main(args)
