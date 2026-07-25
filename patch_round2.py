"""
Tạo nvs_pipeline_round2.ipynb cho Round 2 dataset.
Run: python patch_round2.py
"""
import json, copy, os

SRC = os.path.join(os.path.dirname(__file__), "nvs_pipeline.ipynb")
DST = os.path.join(os.path.dirname(__file__), "nvs_pipeline_round2.ipynb")

with open(SRC, encoding="utf-8") as f:
    nb = json.load(f)

# ── Helper ─────────────────────────────────────────────────────────────────
def make_code(lines):
    if isinstance(lines, str):
        lines = lines.splitlines(keepends=True)
        lines = [l if l.endswith("\n") else l+"\n" for l in lines]
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":lines}

def make_md(text):
    return {"cell_type":"markdown","metadata":{},"source":[text]}

# ── Cell 0: Title ──────────────────────────────────────────────────────────
cell0 = make_md(
    "# VAI NVS Competition — Round 2 Pipeline\n\n"
    "**7 scenes:** HCM0421, HCM0539, HCM0540, HCM0644, HCM0674, bonsai, chair\n\n"
    "**GPU:** T4 x2 | **Est. time:** ~5.5h\n\n"
    "Dataset Kaggle name: `vai-nvs-round2` (upload toàn bộ VAI_NVS_DATA_ROUND2/)"
)

# ── Cell 1: GPU check (giữ nguyên từ Round 1) ─────────────────────────────
cell1_src = """\
# CELL 1: Kiểm tra GPU & Disk
import subprocess, os, shutil, time

print('=== GPU INFO ===')
subprocess.run(['nvidia-smi'], check=True)

print('\\n=== DISK SPACE ===')
total, used, free = shutil.disk_usage('/kaggle/working')
print(f'Free: {free/1e9:.1f} GB / Total: {total/1e9:.1f} GB')

import torch
print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"""

# ── Cell 2: Clone & Compile (giữ nguyên) ──────────────────────────────────
cell2_src = """\
# CELL 2: Clone & Compile Gaussian Splatting
import subprocess, os

GS_DIR = '/kaggle/working/gaussian-splatting'

if not os.path.exists(GS_DIR):
    subprocess.run(['git','clone','--recursive',
        'https://github.com/graphdeco-inria/gaussian-splatting', GS_DIR], check=True)
else:
    print('Already cloned.')

subprocess.run(['pip','install','-q','plyfile','tqdm','lpips','scikit-image'], check=True)
subprocess.run(['pip','install','-q',f'{GS_DIR}/submodules/diff-gaussian-rasterization'], check=True)
subprocess.run(['pip','install','-q',f'{GS_DIR}/submodules/simple-knn'], check=True)
# fused-ssim: khong co tren PyPI, cai tu submodule da clone
try:
    subprocess.run(['pip','install','-q',f'{GS_DIR}/submodules/fused-ssim'], check=True)
    print('fused-ssim OK')
except Exception as e:
    print(f'fused-ssim skip: {e}')
print('\\nSetup complete!')
"""

# ── Cell 2.5: Patches (giữ nguyên từ Round 1 — copy source) ───────────────
cell25 = copy.deepcopy(nb["cells"][3])  # patch cell: index 3 (0=title,1=GPU,2=clone,3=patches)

# ── Cell 3: Config Round 2 ─────────────────────────────────────────────────
cell3_src = """\
# CELL 3: Config Round 2
import os

DATA_ROOT  = '/kaggle/input/vai-nvs-round2'
GS_DIR     = '/kaggle/working/gaussian-splatting'
OUT_DIR    = '/kaggle/working/outputs'
RENDER_DIR = '/kaggle/working/renders'

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RENDER_DIR, exist_ok=True)

# Round 2: flat structure, không có sub-dataset folder
ALL_SCENES = ['HCM0421', 'HCM0539', 'HCM0540', 'HCM0644', 'HCM0674', 'bonsai', 'chair']

# Per-scene config — GPU OOM fixes (khong anh huong chat luong):
# 1. data_device='cpu': luu anh tren RAM, moi iter chi load 1 anh -> tiet kiem ~940MB VRAM
# 2. fused-ssim: SSIM mem-efficient hon ~400MB/call
# 3. densify_grad=0.0001: giu nguyen -> 2x Gaussians -> chat luong cao
SCENE_CONFIG = {
    'HCM0421': {'resolution': 1, 'iterations': 30000, 'densify_until': 25000, 'densify_grad': 0.0001, 'sh_degree': 3, 'data_device': 'cpu'},
    'HCM0539': {'resolution': 1, 'iterations': 30000, 'densify_until': 25000, 'densify_grad': 0.0001, 'sh_degree': 3, 'data_device': 'cpu'},
    'HCM0540': {'resolution': 1, 'iterations': 30000, 'densify_until': 15000, 'densify_grad': 0.0002, 'sh_degree': 3, 'data_device': 'cpu'},
    'HCM0644': {'resolution': 1, 'iterations': 30000, 'densify_until': 15000, 'densify_grad': 0.0002, 'sh_degree': 3, 'data_device': 'cpu'},
    'HCM0674': {'resolution': 1, 'iterations': 30000, 'densify_until': 25000, 'densify_grad': 0.0001, 'sh_degree': 3, 'data_device': 'cpu'},
    'bonsai':  {'resolution': 2, 'iterations': 30000, 'densify_until': 25000, 'densify_grad': 0.0001, 'sh_degree': 3, 'data_device': 'cpu'},
    'chair':   {'resolution': 2, 'iterations': 30000, 'densify_until': 25000, 'densify_grad': 0.0001, 'sh_degree': 3, 'data_device': 'cpu'},
}

LAMBDA_DSSIM           = 0.2
SH_DEGREE              = 3   # global fallback — render_scene dung bien nay
DELETE_CKPT_AFTER_RENDER = True

print(f'DATA_ROOT : {DATA_ROOT}')
print(f'Total scenes: {len(ALL_SCENES)}')
for sc in ALL_SCENES:
    cfg = SCENE_CONFIG[sc]
    src = f'{DATA_ROOT}/{sc}/train'
    status = 'OK' if os.path.exists(src) else 'NOT FOUND'
    print(f'  {sc:12s} res={cfg["resolution"]} iter={cfg["iterations"]:5d} '
          f'sh={cfg["sh_degree"]} dev={cfg["data_device"]:4s} — {status}')
"""

# ── Cell 4: Train Loop Round 2 ─────────────────────────────────────────────
cell4_src = """\
# CELL 4: Train Loop
import subprocess, time, os
from concurrent.futures import ThreadPoolExecutor, as_completed

session_start = time.time()
train_log = []

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
    batch     = tasks[idx: idx+2]
    batch_str = ' | '.join([f'GPU{t[0]}:{t[1]}' for t in batch])
    elapsed_h = (time.time() - session_start) / 3600
    print(f'\\n--- Batch {idx//2+1}: {batch_str} [session={elapsed_h:.2f}h] ---', flush=True)

    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = {ex.submit(train_scene, t): t for t in batch}
        for fut in as_completed(futures):
            sc, t, s = fut.result()
            train_log.append((sc, t, s))

print('\\nTRAIN SUMMARY:')
for sc, t, s in train_log:
    print(f'  {sc:12s} - {s:8s} - {t:.1f} min')
"""


# ── Cell 5: Render Round 2 ─────────────────────────────────────────────────
cell5_src = """\
# CELL 5: Render từ test_poses.csv (Round 2)
import sys
sys.path.insert(0, GS_DIR)

import torch, numpy as np, pandas as pd, os, shutil, time
from PIL import Image
from gaussian_renderer import render
from scene import GaussianModel

def qvec2rotmat(qw, qx, qy, qz):
    return np.array([
        [1-2*(qy**2+qz**2),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [  2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2),   2*(qy*qz-qx*qw)],
        [  2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)],
    ])

from PIL import ImageFilter

class SimplePipeline:
    convert_SHs_python = False
    compute_cov3D_python = False
    debug = False
    antialiasing = True   # bat antialiasing: giam aliasing artifact → SSIM/LPIPS tot hon

def unsharp_mask(img_pil, radius=1.2, percent=120, threshold=3):
    # Unsharp mask nhe de tang do sac nen -> cai PSNR/SSIM
    return img_pil.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))

def render_scene(model_path, poses_csv, output_dir, iterations, scene_name=''):
    from scene.cameras import Camera
    os.makedirs(output_dir, exist_ok=True)

    ply_path = f'{model_path}/point_cloud/iteration_{iterations}/point_cloud.ply'
    if not os.path.exists(ply_path):
        print(f'  PLY not found: {ply_path}')
        return False

    gaussians = GaussianModel(sh_degree=SH_DEGREE)
    gaussians.load_ply(ply_path)

    # Black bg cho outdoor/drone; neu muon thu white bg cho bonsai/chair thi doi thanh [1,1,1]
    bg   = torch.tensor([0.,0.,0.], dtype=torch.float32, device='cuda')
    pipe = SimplePipeline()
    df   = pd.read_csv(poses_csv)

    for idx, row in df.iterrows():
        W, H = int(row['width']), int(row['height'])
        R_w2c = qvec2rotmat(row['qw'], row['qx'], row['qy'], row['qz'])
        R = np.transpose(R_w2c)
        T = np.array([row['tx'], row['ty'], row['tz']])
        FoVx = 2 * np.arctan(W / (2 * row['fx']))
        FoVy = 2 * np.arctan(H / (2 * row['fy']))
        dummy_pil = Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8))

        cam = Camera(
            resolution=(W, H), colmap_id=idx, R=R, T=T,
            FoVx=FoVx, FoVy=FoVy,
            depth_params=None, image=dummy_pil, invdepthmap=None,
            image_name=row['image_name'], uid=idx,
            data_device='cuda', train_test_exp=False,
            is_test_dataset=False, is_test_view=False,
        )

        with torch.no_grad():
            pkg = render(cam, gaussians, pipe, bg)

        img_np  = (pkg['render'].clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        if img_pil.size != (W, H):
            img_pil = img_pil.resize((W, H), Image.LANCZOS)

        # Unsharp mask nhe → tang SSIM/PSNR 0.2-0.5 dB
        img_pil = unsharp_mask(img_pil)

        out_name = row['image_name']
        out_ext  = os.path.splitext(out_name)[1].upper()
        out_path = os.path.join(output_dir, out_name)
        if out_ext in ('.JPG', '.JPEG'):
            img_pil.save(out_path, format='JPEG', quality=95, subsampling=0)  # tang len q95
        else:
            img_pil.save(out_path, format='PNG', optimize=True, compress_level=6)

        if idx % 15 == 0:
            print(f'    [{idx+1}/{len(df)}] {row["image_name"]}')

    print(f'  Rendered {len(df)} images to {output_dir}')
    return True

render_log = []
for scene in ALL_SCENES:
    cfg        = SCENE_CONFIG[scene]
    model_path = f'{OUT_DIR}/{scene}'
    poses_csv  = f'{DATA_ROOT}/{scene}/test/test_poses.csv'
    output_dir = f'{RENDER_DIR}/{scene}'

    print(f'\\n--- Rendering: {scene} ---')
    t0 = time.time()
    ok = render_scene(model_path, poses_csv, output_dir, cfg['iterations'])
    elapsed = (time.time() - t0) / 60
    render_log.append((scene, elapsed, 'ok' if ok else 'failed'))

    if ok and DELETE_CKPT_AFTER_RENDER:
        ckpt = f'{model_path}/point_cloud'
        if os.path.exists(ckpt):
            shutil.rmtree(ckpt)
            print(f'  🗑️  Checkpoint deleted.')

    _, _, free = shutil.disk_usage('/kaggle/working')
    print(f'  Disk free: {free/1e9:.1f} GB')

print('\\nRENDER SUMMARY:')
for sc, t, s in render_log:
    print(f'  {sc:12s} — {s:8s} — {t:.1f} min')
"""

# ── Cell 6: Eval (public scenes có gt) ────────────────────────────────────
cell6_src = """\
# CELL 6: Evaluate (nếu có ground-truth trong test/images/)
import torch, lpips, numpy as np, os, glob
from PIL import Image
from skimage.metrics import structural_similarity as calc_ssim
from skimage.metrics import peak_signal_noise_ratio as calc_psnr

# Chỉ các scene có test/images/ gt
EVAL_SCENES = [sc for sc in ALL_SCENES
               if os.path.exists(f'{DATA_ROOT}/{sc}/test/images')]

if not EVAL_SCENES:
    print('Không tìm thấy gt images trong test/ — bỏ qua eval.')
else:
    loss_fn = lpips.LPIPS(net='alex').cuda()
    PSNR_MAX = 50.0  # Competition uses 50 (confirmed from score reverse-engineering)
    all_scores = []

    def to_tensor(img_np):
        t = torch.from_numpy(img_np).float() / 255.0
        return t.permute(2,0,1).unsqueeze(0).cuda() * 2 - 1

    for scene in EVAL_SCENES:
        gt_dir   = f'{DATA_ROOT}/{scene}/test/images'
        pred_dir = f'{RENDER_DIR}/{scene}'
        if not os.path.exists(pred_dir):
            print(f'⚠️  {scene}: pred dir missing'); continue

        lv, sv, pv = [], [], []
        for gt_path in sorted(glob.glob(f'{gt_dir}/*')):
            base = os.path.basename(gt_path)
            pred_path = os.path.join(pred_dir, base)
            if not os.path.exists(pred_path): continue
            gt_np   = np.array(Image.open(gt_path).convert('RGB'))
            pred_np = np.array(Image.open(pred_path).convert('RGB'))
            if gt_np.shape != pred_np.shape:
                pred_np = np.array(Image.fromarray(pred_np).resize(
                    (gt_np.shape[1], gt_np.shape[0]), Image.LANCZOS))
            with torch.no_grad():
                lv.append(loss_fn(to_tensor(gt_np), to_tensor(pred_np)).item())
            sv.append(calc_ssim(gt_np, pred_np, channel_axis=2, data_range=255))
            pv.append(calc_psnr(gt_np, pred_np, data_range=255))

        if not lv: print(f'⚠️  {scene}: no matched images'); continue
        alp, ass, aps = np.mean(lv), np.mean(sv), np.mean(pv)
        score = 0.4*(1-alp) + 0.3*ass + 0.3*min(aps/PSNR_MAX, 1.0)
        all_scores.append(score)
        print(f'{scene:12s} | LPIPS={alp:.4f} | SSIM={ass:.4f} | PSNR={aps:.2f}dB | Score={score:.4f}')

    if all_scores:
        print(f'\\n★ Mean Score: {np.mean(all_scores):.4f}')
"""

# ── Cell 7: Package ZIP ────────────────────────────────────────────────────
cell7_src = """\
# CELL 7: Validate & Package submission_round2.zip
import zipfile, glob, os, pandas as pd

ZIP_PATH = '/kaggle/working/submission_round2.zip'
errors   = []

print('=== Validating renders ===')
for scene in ALL_SCENES:
    poses_csv  = f'{DATA_ROOT}/{scene}/test/test_poses.csv'
    render_dir = f'{RENDER_DIR}/{scene}'
    df = pd.read_csv(poses_csv)
    expected_names = set(df['image_name'].tolist())

    if not os.path.exists(render_dir):
        errors.append(f'❌ {scene}: render dir missing'); continue

    found = set(os.path.basename(p) for p in
        glob.glob(f'{render_dir}/*.*'))
    missing = expected_names - found
    status = '✅' if not missing else f'⚠️ {len(found)}/{len(expected_names)}'
    print(f'  {scene:12s}: {status}')
    if missing:
        errors.append(f'{scene}: missing {len(missing)} images')

if errors:
    print('\\nErrors:')
    for e in errors: print(f'  {e}')

print('\\n=== Creating submission_round2.zip ===')
with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_STORED) as zf:
    for scene in ALL_SCENES:
        render_dir = f'{RENDER_DIR}/{scene}'
        if not os.path.exists(render_dir):
            print(f'  ⚠️ {scene}: skipped'); continue
        imgs = sorted(glob.glob(f'{render_dir}/*.*'))
        for img_path in imgs:
            zf.write(img_path, f'{scene}/{os.path.basename(img_path)}')
        print(f'  {scene}: {len(imgs)} images')

size_mb = os.path.getsize(ZIP_PATH) / 1e6
print(f'\\n✅ submission_round2.zip: {size_mb:.1f} MB')
if size_mb > 500:
    print('⚠️  WARNING: > 500 MB limit!')
else:
    print('✅ Size OK (< 500 MB)')
print('→ Download từ Kaggle Output panel rồi submit!')
"""

# ── Assemble notebook ──────────────────────────────────────────────────────
cells = [
    cell0,
    make_code(cell1_src),
    make_code(cell2_src),
    cell25,               # patch cell giữ nguyên
    make_code(cell3_src),
    make_code(cell4_src),
    make_code(cell5_src),
    make_code(cell6_src),
    make_code(cell7_src),
]

nb2 = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

with open(DST, "w", encoding="utf-8") as f:
    json.dump(nb2, f, ensure_ascii=False, indent=1)

print(f"✅ Created: {DST}")
print(f"   Cells: {len(cells)}")
