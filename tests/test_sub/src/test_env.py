from env_wrapper import PokemonTCGEnv, read_sample_deck

def test_env():
    deck = read_sample_deck()
    env = PokemonTCGEnv(deck, deck)
    obs, info = env.reset()
    print("Reset successful. Action mask shape:", obs["action_mask"].shape)
    
    done = False
    steps = 0
    while not done and steps < 10:
        # Sample an action uniformly from the valid actions
        valid_actions = [i for i, mask in enumerate(obs["action_mask"]) if mask == 1]
        action = valid_actions[0] if valid_actions else 0
        
        obs, reward, done, truncated, info = env.step(action)
        steps += 1
        print(f"Step {steps} complete. Done: {done}, Reward: {reward}")
    
    env.close()
    print("Test finished successfully!")

if __name__ == "__main__":
    test_env()
