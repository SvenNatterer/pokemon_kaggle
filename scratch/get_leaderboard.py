from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

leaderboard = api.competition_leaderboard_view("pokemon-tcg-ai-battle")
print(f"Leaderboard type: {type(leaderboard)}")
print(f"Number of items: {len(leaderboard)}")
if len(leaderboard) > 0:
    first = leaderboard[0]
    print("Leaderboard item attributes:")
    for attr in dir(first):
        if not attr.startswith('_'):
            try:
                print(f"  {attr}: {getattr(first, attr)}")
            except Exception as e:
                print(f"  {attr}: Error: {e}")
