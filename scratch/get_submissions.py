import json
import sys
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

subs = api.competition_submissions("pokemon-tcg-ai-battle")
print(f"Found {len(subs)} submissions")
if len(subs) > 0:
    sub = subs[0]
    print("Attributes of submission object:")
    for attr in dir(sub):
        if not attr.startswith('_'):
            try:
                print(f"  {attr}: {getattr(sub, attr)}")
            except Exception as e:
                print(f"  {attr}: Error reading: {e}")
