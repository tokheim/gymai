import logging
import random

log = logging.getLogger(__name__)

class NoopGameHacks(object):
    def __init__(self):
        pass

    def on_start(self, env, state):
        return state

    def should_end(self, info):
        return False

    def custom_reward(self, info):
        return 0

    def on_end(self, memories):
        pass

class BreakoutHacks(NoopGameHacks):
    def __init__(self):
        self.lives = -1

    def on_start(self, env, orig_state):
        for i in range(random.randrange(0, 20)):
            action = random.randrange(0,3)
            if action > 0:
                action += 1
            env.step(action)#randomize start
        state, _, _, _  = env.step(1)#release ball
        return state

    def custom_reward(self, info):
        lives = info.get('ale.lives', self.lives)
        reward = 0
        if lives < self.lives:
            #reward = -1
            pass
        self.lives = lives
        return reward

    def should_end(self, info):
        return False
        #return info.get('ale.lives', None) < 5

class CartpoleHacks(NoopGameHacks):
    def __init__(self):
        pass

    def on_end(self, memories):
        if len(memories) > 499:
            memories[-1].reward = 0

def load_hacks(envname):
    if envname in _hack_map:
        cls = _hack_map[envname]
        log.info("Using env mods from "+str(cls))
        return cls()
    return NoopGameHacks()

_hack_map = {
        "Breakout-v0": BreakoutHacks,
        "CartPole-v1": CartpoleHacks
}
