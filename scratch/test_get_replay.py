import urllib.request
import json

url = "https://www.kaggle.com/requests/EpisodeService/GetEpisodeReplay"
payload = json.dumps({"EpisodeId": 87037854}).encode('utf-8')

req = urllib.request.Request(
    url, 
    data=payload, 
    headers={'Content-Type': 'application/json'}
)

try:
    with urllib.request.urlopen(req) as response:
        html = response.read().decode('utf-8')
        data = json.loads(html)
        print("Success! JSON Keys:")
        print(list(data.keys()))
        if 'agents' in data:
            print("Agents:")
            for a in data['agents']:
                print(a)
        else:
            print("Preview of data:", str(data)[:1000])
except Exception as e:
    print("Error:", e)
