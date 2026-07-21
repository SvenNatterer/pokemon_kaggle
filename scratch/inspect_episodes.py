import requests
import datetime

url = 'https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes'
sub_ids = [54834368, 54815682, 54793285, 54784364, 54723890]

def parse_time(t_str):
    t_str = t_str.replace('Z', '')
    if '.' in t_str:
        parts = t_str.split('.')
        fraction = parts[1][:6]
        t_str = parts[0] + '.' + fraction
    return datetime.datetime.fromisoformat(t_str)

for sub_id in sub_ids:
    r = requests.post(url, json={'submissionId': sub_id})
    if r.status_code != 200:
        print(f"Failed to fetch {sub_id}")
        continue
    data = r.json()
    episodes = data.get('episodes', [])
    if not episodes:
        print(f"No episodes for {sub_id}")
        continue
    
    times = []
    for ep in episodes:
        times.append(parse_time(ep['createTime']))
    times.sort()
    
    if times:
        duration = times[-1] - times[0]
        print(f"Sub {sub_id}: {len(episodes)} games, Start: {times[0]}, End: {times[-1]}, Duration: {duration}")
    else:
        print(f"Sub {sub_id}: {len(episodes)} games, No times")
