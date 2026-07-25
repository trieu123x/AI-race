import json
with open('nvs_pipeline_round2.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    n = len(c['source'])
    first = (c['source'][0] if c['source'] else '').encode('ascii','replace').decode()[:55].strip()
    print(f'Cell {i} [{c["cell_type"]}] ({n} lines): {first}')

src3 = ''.join(nb['cells'][3]['source'])
print()
for kw in ['SIMPLE_RADIAL', 'storePly', 'filter None cameras', 'save_ply memory']:
    print(f'  Patch [{kw}]: {kw in src3}')
