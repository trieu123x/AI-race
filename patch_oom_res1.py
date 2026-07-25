"""
patch_oom_res1_v2.py
====================
Fix OOM cho HCM0540/HCM0644 giu res=1:
  - densify_grad:  0.0001 -> 0.0002  (paper default)
  - densify_until: 25000  -> 15000   (dung densify som, gia VRAM)
"""
import json

NB_PATH = r"c:\Users\admin\Downloads\VAI_NVS_DATA\nvs_pipeline_round2.ipynb"

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Exact lines lay tu repr() o tren
OLD_540 = "    'HCM0540': {'resolution': 1, 'iterations': 30000, 'densify_until': 25000, 'densify_grad': 0.0001, 'sh_degree': 3, 'data_device': 'cpu'},\n"
NEW_540 = "    'HCM0540': {'resolution': 1, 'iterations': 30000, 'densify_until': 15000, 'densify_grad': 0.0002, 'sh_degree': 3, 'data_device': 'cpu'},\n"

OLD_644 = "    'HCM0644': {'resolution': 1, 'iterations': 30000, 'densify_until': 25000, 'densify_grad': 0.0001, 'sh_degree': 3, 'data_device': 'cpu'},\n"
NEW_644 = "    'HCM0644': {'resolution': 1, 'iterations': 30000, 'densify_until': 15000, 'densify_grad': 0.0002, 'sh_degree': 3, 'data_device': 'cpu'},\n"

patched = 0
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    new_src = []
    for line in cell["source"]:
        if line == OLD_540:
            new_src.append(NEW_540)
            patched += 1
            print(f"  Patched HCM0540")
        elif line == OLD_644:
            new_src.append(NEW_644)
            patched += 1
            print(f"  Patched HCM0644")
        else:
            new_src.append(line)
    cell["source"] = new_src

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"\nTotal patched: {patched} line(s)")
print("\nVerify SCENE_CONFIG:")
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        for line in cell["source"]:
            if "'HCM" in line or "'bonsai'" in line or "'chair'" in line:
                if "densify" in line:
                    print(" ", line.rstrip())
