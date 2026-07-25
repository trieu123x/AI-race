"""
patch_oom_notqdm.py
===================
1. Patch train.py  : bo tqdm, thay bang print moi 500 iter
2. Patch notebook  : tang resolution HCM0540/HCM0644 len 2 de fix OOM
"""
import json, re, sys, os

# ──────────────────────────────────────────────
# 1. PATCH train.py
# ──────────────────────────────────────────────
TRAIN_PY = r"c:\Users\admin\Downloads\VAI_NVS_DATA\gaussian-splatting\train.py"

with open(TRAIN_PY, "r", encoding="utf-8") as f:
    src = f.read()

# 1a: Bo import tqdm
src = src.replace("from tqdm import tqdm\n", "# tqdm removed\n")

# 1b: Bo khoi tao progress_bar
OLD_PBAR_INIT = '    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")\n'
NEW_PBAR_INIT = '    _log_every = 500  # print loss every N iterations\n'
src = src.replace(OLD_PBAR_INIT, NEW_PBAR_INIT)

# 1c: Thay 3 dong progress_bar.set_postfix / update / close
OLD_PBAR_BLOCK = (
    '            if iteration % 10 == 0:\n'
    '                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})\n'
    '                progress_bar.update(10)\n'
    '            if iteration == opt.iterations:\n'
    '                progress_bar.close()\n'
)
NEW_PBAR_BLOCK = (
    '            if iteration % _log_every == 0 or iteration == opt.iterations:\n'
    '                print(f"[{iteration}/{opt.iterations}] Loss={ema_loss_for_log:.7f}", flush=True)\n'
)
src = src.replace(OLD_PBAR_BLOCK, NEW_PBAR_BLOCK)

with open(TRAIN_PY, "w", encoding="utf-8") as f:
    f.write(src)

# Kiem tra
checks = [
    ("tqdm removed",           "from tqdm import tqdm" not in src),
    ("_log_every defined",     "_log_every" in src),
    ("progress_bar removed",   "progress_bar" not in src),
]
print("=== train.py patches ===")
for name, ok in checks:
    print(f"  {'OK' if ok else 'FAIL'}: {name}")


# ──────────────────────────────────────────────
# 2. PATCH notebook: fix OOM config
# ──────────────────────────────────────────────
NB_PATH = r"c:\Users\admin\Downloads\VAI_NVS_DATA\nvs_pipeline_round2.ipynb"

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Tim cell CELL 3 (SCENE_CONFIG) va thay resolution + densify_grad cho cac scene nang
# HCM0540 va HCM0644: res 1->2, densify_grad 0.0001->0.00015 (it Gaussian hon ~30%)
# HCM0421, HCM0539, HCM0674: giu res=1 nhung tang densify_grad len 0.00015

fixes = {
    # (old_line_fragment, new_line_fragment)
    "'HCM0540': {'resolution': 1,": "'HCM0540': {'resolution': 2,",
    "'HCM0644': {'resolution': 1,": "'HCM0644': {'resolution': 2,",
    # Tang densify_grad cho tat ca scene res=1 de it Gaussian hon
    "'HCM0421': {'resolution': 1, 'iterations': 30000, 'densify_until': 25000, 'densify_grad': 0.0001,":
        "'HCM0421': {'resolution': 1, 'iterations': 30000, 'densify_until': 25000, 'densify_grad': 0.00015,",
    "'HCM0539': {'resolution': 1, 'iterations': 30000, 'densify_until': 25000, 'densify_grad': 0.0001,":
        "'HCM0539': {'resolution': 1, 'iterations': 30000, 'densify_until': 25000, 'densify_grad': 0.00015,",
    "'HCM0674': {'resolution': 1, 'iterations': 30000, 'densify_until': 25000, 'densify_grad': 0.0001,":
        "'HCM0674': {'resolution': 1, 'iterations': 30000, 'densify_until': 25000, 'densify_grad': 0.00015,",
    # HCM0540/0644 cung tang densify_grad
    "'HCM0540': {'resolution': 2, 'iterations': 30000, 'densify_until': 25000, 'densify_grad': 0.0001,":
        "'HCM0540': {'resolution': 2, 'iterations': 30000, 'densify_until': 25000, 'densify_grad': 0.00015,",
    "'HCM0644': {'resolution': 2, 'iterations': 30000, 'densify_until': 25000, 'densify_grad': 0.0001,":
        "'HCM0644': {'resolution': 2, 'iterations': 30000, 'densify_until': 25000, 'densify_grad': 0.00015,",
}

nb_patched = 0
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    new_src = []
    for line in cell["source"]:
        replaced = line
        for old_frag, new_frag in fixes.items():
            if old_frag in line:
                replaced = line.replace(old_frag, new_frag)
                nb_patched += 1
                break
        new_src.append(replaced)
    cell["source"] = new_src

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"\n=== notebook patches ===")
print(f"  {nb_patched} line(s) patched in SCENE_CONFIG")

print("\nDone. Summary:")
print("  train.py  : tqdm removed, print every 500 iter")
print("  notebook  : HCM0540/0644 resolution 1->2")
print("  notebook  : densify_grad 0.0001->0.00015 for all res=1 scenes")
