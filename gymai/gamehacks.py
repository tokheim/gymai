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

    def on_end(self, memories):
        pass

class BreakoutHacks(NoopGameHacks):
    def __init__(self):
        pass

    def on_start(self, env, orig_state):
        for i in range(random.randrange(0, 20)):
            env.step(0)#randomize start
        state, _, _, _  = env.step(1)#release ball
        return state

    def should_end(self, info):
        return info.get('ale.lives') == 4

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
