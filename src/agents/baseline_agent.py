import numpy as np

class RandomAgent:
    """A baseline agent that chooses a random valid action."""
    def __init__(self, env):
        self.env = env
        
    def predict(self, obs, deterministic=False):
        # In our env wrapper, obs is a dict with 'action_mask'
        mask = obs['action_mask']
        valid_actions = np.where(mask == 1)[0]
        if len(valid_actions) == 0:
            return 0, None
        action = np.random.choice(valid_actions)
        return action, None
