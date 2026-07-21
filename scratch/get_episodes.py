from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

# Let's list episodes for submission 54834368
episodes = api.competition_list_episodes(54834368)
print(f"Found {len(episodes)} episodes")
if len(episodes) > 0:
    ep = episodes[0]
    print("Attributes of episode object:")
    for attr in dir(ep):
        if not attr.startswith('_'):
            try:
                print(f"  {attr}: {getattr(ep, attr)}")
            except Exception as e:
                print(f"  {attr}: Error: {e}")
                
    # Check agents
    print("Agents in the episode:")
    for agent in getattr(ep, 'agents', []):
        print(f"  Agent ref: {getattr(agent, 'ref', None)}")
        print(f"  Submission ID: {getattr(agent, 'submission_id', None)}")
        print(f"  Updated Score: {getattr(agent, 'updated_score', None)}")
        print(f"  Updated Rank: {getattr(agent, 'updated_rank', None)}")
        print(f"  Index: {getattr(agent, 'index', None)}")
        print(f"  Initial Score: {getattr(agent, 'initial_score', None)}")
        # Check print of agent attributes
        print("  All agent attributes:")
        for attr in dir(agent):
            if not attr.startswith('_'):
                try:
                    print(f"    {attr}: {getattr(agent, attr)}")
                except Exception as ae:
                    print(f"    {attr}: Error: {ae}")
