"""
patch_cell4_no_monitor.py
=========================
Xoa monitor_loop / parse_progress / threading khoi Cell 4
Giu nguyen train_scene va batch loop
"""
import json

NB_PATH = r"c:\Users\admin\Downloads\VAI_NVS_DATA\nvs_pipeline_round2.ipynb"

NEW_CELL4_SOURCE = [
    "# CELL 4: Train Loop\n",
    "import subprocess, time, os\n",
    "from concurrent.futures import ThreadPoolExecutor, as_completed\n",
    "\n",
    "session_start = time.time()\n",
    "train_log = []\n",
    "\n",
    "def train_scene(args):\n",
    "    gpu_id, scene = args\n",
    "    cfg        = SCENE_CONFIG[scene]\n",
    "    iterations = cfg['iterations']\n",
    "    src_path   = f'{DATA_ROOT}/{scene}/train'\n",
    "    model_path = f'{OUT_DIR}/{scene}'\n",
    "    log_path   = f'{model_path}/train.log'\n",
    "    ply_path   = f'{model_path}/point_cloud/iteration_{iterations}/point_cloud.ply'\n",
    "\n",
    "    if os.path.exists(ply_path):\n",
    "        print(f'[GPU{gpu_id}] {scene}: checkpoint found, skipping.', flush=True)\n",
    "        return scene, 0, 'skipped'\n",
    "\n",
    "    os.makedirs(model_path, exist_ok=True)\n",
    "    env = os.environ.copy()\n",
    "    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)\n",
    "    env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'\n",
    "    env['PYTHONUNBUFFERED'] = '1'\n",
    "\n",
    "    t0 = time.time()\n",
    "    print(f'[GPU{gpu_id}] {scene}: starting  (log: {log_path})', flush=True)\n",
    "\n",
    "    with open(log_path, 'w', buffering=1) as lf:\n",
    "        proc = subprocess.Popen([\n",
    "            'python', f'{GS_DIR}/train.py',\n",
    "            '-s', src_path, '-m', model_path,\n",
    "            '--iterations',             str(iterations),\n",
    "            '--sh_degree',              str(cfg['sh_degree']),\n",
    "            '--densify_until_iter',     str(cfg['densify_until']),\n",
    "            '--densify_grad_threshold', str(cfg['densify_grad']),\n",
    "            '--lambda_dssim',           str(LAMBDA_DSSIM),\n",
    "            '--save_iterations',        str(iterations),\n",
    "            '--test_iterations',        '-1',\n",
    "            '--resolution',             str(cfg['resolution']),\n",
    "            '--position_lr_max_steps',  str(iterations),\n",
    "            '--disable_viewer',\n",
    "            '--data_device',            cfg['data_device'],\n",
    "        ], stdout=lf, stderr=subprocess.STDOUT, env=env, text=True, bufsize=1)\n",
    "        proc.wait()\n",
    "\n",
    "    elapsed_min = (time.time() - t0) / 60\n",
    "    if proc.returncode != 0:\n",
    "        print(f'[GPU{gpu_id}] {scene} FAILED after {elapsed_min:.1f} min', flush=True)\n",
    "        with open(log_path) as lf:\n",
    "            print(lf.read()[-2000:], flush=True)\n",
    "        return scene, elapsed_min, 'failed'\n",
    "\n",
    "    print(f'[GPU{gpu_id}] {scene} DONE  {elapsed_min:.1f} min', flush=True)\n",
    "    return scene, elapsed_min, 'ok'\n",
    "\n",
    "tasks = [(i % 2, sc) for i, sc in enumerate(ALL_SCENES)]\n",
    "print(f'Running {len(tasks)} scenes on 2 GPUs...', flush=True)\n",
    "\n",
    "for idx in range(0, len(tasks), 2):\n",
    "    batch     = tasks[idx: idx+2]\n",
    "    batch_str = ' | '.join([f'GPU{t[0]}:{t[1]}' for t in batch])\n",
    "    elapsed_h = (time.time() - session_start) / 3600\n",
    "    print(f'\\n--- Batch {idx//2+1}: {batch_str} [session={elapsed_h:.2f}h] ---', flush=True)\n",
    "\n",
    "    with ThreadPoolExecutor(max_workers=2) as ex:\n",
    "        futures = {ex.submit(train_scene, t): t for t in batch}\n",
    "        for fut in as_completed(futures):\n",
    "            sc, t, s = fut.result()\n",
    "            train_log.append((sc, t, s))\n",
    "\n",
    "print('\\nTRAIN SUMMARY:')\n",
    "for sc, t, s in train_log:\n",
    "    print(f'  {sc:12s} - {s:8s} - {t:.1f} min')\n",
]

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

patched = False
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        if "monitor_loop" in src or "CELL 4" in src:
            cell["source"] = NEW_CELL4_SOURCE
            patched = True
            print(f"Patched cell index {i}")
            break

if not patched:
    print("ERROR: Cell 4 not found!")
else:
    with open(NB_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("Done. monitor_loop, parse_progress, threading removed from Cell 4.")
