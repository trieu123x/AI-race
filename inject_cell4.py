"""
Patch cell4_src trong patch_round2.py: thay toan bo cell4_src cu bang version moi co progress monitor.
Run: python inject_cell4.py
"""
import re, os

TARGET = os.path.join(os.path.dirname(__file__), "patch_round2.py")

NEW_CELL4 = r'''# ── Cell 4: Train Loop Round 2 ─────────────────────────────────────────────
cell4_src = """\
# CELL 4: Train Loop — progress monitor every 30s
import subprocess, time, os, re, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

session_start = time.time()
train_log = []

def parse_progress(log_path, total_iter):
    if not os.path.exists(log_path):
        return None
    try:
        with open(log_path, 'rb') as f:
            f.seek(max(0, os.path.getsize(log_path) - 3000))
            tail = f.read().decode('utf-8', errors='ignore')
        hits = re.findall(
            r'(\d+)/\d+\s+\[[\d:]+<([\d:]+),\s*([\d.]+)it/s[^\n]*Loss=([\d.eE+-]+)',
            tail)
        if hits:
            cur, eta, spd, loss = hits[-1]
            pct  = int(cur) * 100 // total_iter
            done = pct // 5
            bar  = '=' * done + '-' * (20 - done)
            return f'[{bar}] {pct:3d}%  {cur}/{total_iter}  {spd}it/s  Loss={loss}  ETA={eta}'
        if os.path.getsize(log_path) > 0:
            return '[--------------------]   0%  loading scene...'
    except Exception:
        pass
    return None

def monitor_loop(batch_scenes, stop_evt):
    while not stop_evt.wait(30):
        lines = [f'\\n  -- PROGRESS {time.strftime("%H:%M:%S")} --']
        for sc in batch_scenes:
            p = parse_progress(f'{OUT_DIR}/{sc}/train.log', SCENE_CONFIG[sc]['iterations'])
            lines.append(f'  {sc:12s}: {p or "waiting..."}')
        elapsed_h = (time.time() - session_start) / 3600
        lines.append(f'  Session: {elapsed_h:.2f}h / 12h')
        print('\\n'.join(lines), flush=True)

def train_scene(args):
    gpu_id, scene = args
    cfg        = SCENE_CONFIG[scene]
    iterations = cfg['iterations']
    src_path   = f'{DATA_ROOT}/{scene}/train'
    model_path = f'{OUT_DIR}/{scene}'
    log_path   = f'{model_path}/train.log'
    ply_path   = f'{model_path}/point_cloud/iteration_{iterations}/point_cloud.ply'

    if os.path.exists(ply_path):
        print(f'[GPU{gpu_id}] {scene}: checkpoint found, skipping.', flush=True)
        return scene, 0, 'skipped'

    os.makedirs(model_path, exist_ok=True)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    env['PYTHONUNBUFFERED'] = '1'

    t0 = time.time()
    print(f'[GPU{gpu_id}] {scene}: starting  (log: {log_path})', flush=True)

    with open(log_path, 'w', buffering=1) as lf:
        proc = subprocess.Popen([
            'python', f'{GS_DIR}/train.py',
            '-s', src_path, '-m', model_path,
            '--iterations',             str(iterations),
            '--sh_degree',              str(cfg['sh_degree']),
            '--densify_until_iter',     str(cfg['densify_until']),
            '--densify_grad_threshold', str(cfg['densify_grad']),
            '--lambda_dssim',           str(LAMBDA_DSSIM),
            '--save_iterations',        str(iterations),
            '--test_iterations',        '-1',
            '--resolution',             str(cfg['resolution']),
            '--position_lr_max_steps',  str(iterations),
            '--disable_viewer',
            '--data_device',            cfg['data_device'],
        ], stdout=lf, stderr=subprocess.STDOUT, env=env, text=True, bufsize=1)
        proc.wait()

    elapsed_min = (time.time() - t0) / 60
    if proc.returncode != 0:
        print(f'[GPU{gpu_id}] {scene} FAILED after {elapsed_min:.1f} min', flush=True)
        with open(log_path) as lf:
            print(lf.read()[-2000:], flush=True)
        return scene, elapsed_min, 'failed'

    print(f'[GPU{gpu_id}] {scene} DONE  {elapsed_min:.1f} min', flush=True)
    return scene, elapsed_min, 'ok'

tasks = [(i % 2, sc) for i, sc in enumerate(ALL_SCENES)]
print(f'Running {len(tasks)} scenes on 2 GPUs...', flush=True)

for idx in range(0, len(tasks), 2):
    batch      = tasks[idx: idx+2]
    batch_sc   = [t[1] for t in batch]
    elapsed_h  = (time.time() - session_start) / 3600
    batch_str  = ' | '.join([f'GPU{t[0]}:{t[1]}' for t in batch])
    print(f'\\n--- Batch {idx//2+1}: {batch_str} [session={elapsed_h:.2f}h] ---', flush=True)

    stop_evt = threading.Event()
    mon = threading.Thread(target=monitor_loop, args=(batch_sc, stop_evt), daemon=True)
    mon.start()

    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = {ex.submit(train_scene, t): t for t in batch}
        for fut in as_completed(futures):
            sc, t, s = fut.result()
            train_log.append((sc, t, s))

    stop_evt.set()
    mon.join(timeout=2)

print('\\nTRAIN SUMMARY:')
for sc, t, s in train_log:
    print(f'  {sc:12s} - {s:8s} - {t:.1f} min')
"""

'''

with open(TARGET, encoding='utf-8') as f:
    code = f.read()

# Find and replace the entire cell4_src block
pattern = r'(# ── Cell 4: Train Loop Round 2 ─+\ncell4_src = """\\.*?""")\s*\n'
m = re.search(pattern, code, re.DOTALL)
if m:
    code = code[:m.start()] + NEW_CELL4 + '\n' + code[m.end():]
    with open(TARGET, 'w', encoding='utf-8') as f:
        f.write(code)
    print("OK: cell4_src replaced successfully")
    print(f"New length: {len(code)} chars")
else:
    print("ERROR: pattern not found")
    # Debug: show where cell4_src starts
    idx = code.find('cell4_src')
    if idx >= 0:
        print(f"Found cell4_src at char {idx}")
        print(repr(code[idx:idx+100]))
