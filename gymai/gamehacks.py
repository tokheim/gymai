class NoopGameHacks(object):
    def __init__(self):
        pass

    def on_start(self, env, state):
        return state

    def should_end(self, info):
        return False

class BreakoutHacks(NoopGameHacks):
    def __init__(self):
        pass

    def on_start(self, env, orig_state):
        state, _, _, _  = env.step(1)#release ball
        return state

    def should_end(self, info):
        return info.get('ale.lives') == 4

def load_hacks(envname):
    if envname in _hack_map:
        return _hack_map[envname]()
    return NoopGameHacks()

_hack_map = {
        "Breakout-v0": BreakoutHacks
}
