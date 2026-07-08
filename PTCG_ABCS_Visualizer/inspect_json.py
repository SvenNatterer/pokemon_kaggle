import json
with open('samples/MegaLucario_vs_MegaAbomasnow.json', 'r') as f:
    data = json.load(f)
for snap in data:
    for p in snap['current']['players']:
        for b in (p.get('bench', []) + p.get('active', [])):
            if b.get('energies') or b.get('energyCards'):
                print("Energies:", b.get('energies'), b.get('energyCards'))
                exit(0)
