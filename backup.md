# CELL 1: Kiểm tra GPU & Disk
import subprocess, os, shutil, time

print('=== GPU INFO ===')
subprocess.run(['nvidia-smi'], check=True)

print('\n=== DISK SPACE ===')
total, used, free = shutil.disk_usage('/kaggle/working')
print(f'Free: {free/1e9:.1f} GB / Total: {total/1e9:.1f} GB')

import torch
print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')

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
print('\n✅ Setup complete!')
# ============================================================
# CELL 2: Clone & Compile Gaussian Splatting
# (Internet phải BẬT — chạy 1 lần, mất ~10-15 phút)
# ============================================================
import subprocess, os

GS_DIR = '/kaggle/working/gaussian-splatting'

if not os.path.exists(GS_DIR):
    print('Cloning gaussian-splatting...')
    subprocess.run([
        'git', 'clone', '--recursive',
        'https://github.com/graphdeco-inria/gaussian-splatting',
        GS_DIR
    ], check=True)
else:
    print('gaussian-splatting already cloned.')

print('\nInstalling Python dependencies...')
subprocess.run(['pip', 'install', '-q', 'plyfile', 'tqdm', 'lpips', 'scikit-image'], check=True)

print('\nCompiling diff-gaussian-rasterization (mất ~5-8 phút)...')
subprocess.run(
    ['pip', 'install', '-q', f'{GS_DIR}/submodules/diff-gaussian-rasterization'],
    check=True
)

print('Compiling simple-knn...')
subprocess.run(
    ['pip', 'install', '-q', f'{GS_DIR}/submodules/simple-knn'],
    check=True
)

print('\n✅ Setup complete!')
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

# Per-scene config (resolution & iterations)
SCENE_CONFIG = {
    # BTS drone: ảnh đã scale 1/4 → dùng resolution 1 (không downsample thêm)
    'HCM0421': {'resolution': 1, 'iterations': 20000, 'densify_until': 15000},
    'HCM0539': {'resolution': 1, 'iterations': 20000, 'densify_until': 15000},
    'HCM0540': {'resolution': 1, 'iterations': 20000, 'densify_until': 15000},
    'HCM0644': {'resolution': 1, 'iterations': 20000, 'densify_until': 15000},
    'HCM0674': {'resolution': 1, 'iterations': 20000, 'densify_until': 15000},
    # Object-centric / indoor: full-res hoặc 1.5x scale → cần iter nhiều hơn
    'bonsai':  {'resolution': 2, 'iterations': 30000, 'densify_until': 20000},
    'chair':   {'resolution': 2, 'iterations': 30000, 'densify_until': 20000},
}

SH_DEGREE              = 3
LAMBDA_DSSIM           = 0.2
DELETE_CKPT_AFTER_RENDER = True

print(f'DATA_ROOT : {DATA_ROOT}')
print(f'Total scenes: {len(ALL_SCENES)}')
for sc in ALL_SCENES:
    cfg = SCENE_CONFIG[sc]
    src = f'{DATA_ROOT}/{sc}/train'
    status = '✅' if os.path.exists(src) else '❌ NOT FOUND'
    print(f'  {sc:12s} res={cfg["resolution"]} iter={cfg["iterations"]:5d} — {status}')
# CELL 4: Train Loop Round 2 (2 GPUs parallel)
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
    ply_path   = f'{model_path}/point_cloud/iteration_{iterations}/point_cloud.ply'

    if os.path.exists(ply_path):
        print(f'[GPU{gpu_id}] ⏭️  {scene} checkpoint found, skipping.')
        return scene, 0, 'skipped'

    os.makedirs(model_path, exist_ok=True)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    t0 = time.time()
    result = subprocess.run([
        'python', f'{GS_DIR}/train.py',
        '-s', src_path,
        '-m', model_path,
        '--iterations',         str(iterations),
        '--sh_degree',          str(SH_DEGREE),
        '--densify_until_iter', str(cfg['densify_until']),
        '--lambda_dssim',       str(LAMBDA_DSSIM),
        '--save_iterations',    str(iterations),
        '--test_iterations',    '-1',
        '--resolution',         str(cfg['resolution']),
        '--quiet',
        '--disable_viewer',
        '--data_device',        'cpu',
    ], capture_output=True, text=True, env=env)

    elapsed_min = (time.time() - t0) / 60
    if result.returncode != 0:
        print(f'[GPU{gpu_id}] ❌ {scene} FAILED after {elapsed_min:.1f} min')
        print(result.stderr[-1500:])
        return scene, elapsed_min, 'failed'
    print(f'[GPU{gpu_id}] ✅ {scene} done in {elapsed_min:.1f} min')
    return scene, elapsed_min, 'ok'

tasks = [(i % 2, sc) for i, sc in enumerate(ALL_SCENES)]

print(f'Running {len(tasks)} scenes on 2 GPUs...')
for idx in range(0, len(tasks), 2):
    batch = tasks[idx: idx+2]
    elapsed_h = (time.time() - session_start) / 3600
    batch_str = ' | '.join([f'GPU{t[0]}:{t[1]}' for t in batch])
    print(f'--- Batch {idx//2+1}: {batch_str} [elapsed: {elapsed_h:.1f}h] ---')
    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = {ex.submit(train_scene, t): t for t in batch}
        for fut in as_completed(futures):
            sc, t, s = fut.result()
            train_log.append((sc, t, s))

print('\nTRAIN SUMMARY:')
for sc, t, s in train_log:
    print(f'  {sc:12s} — {s:8s} — {t:.1f} min')
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

class SimplePipeline:
    convert_SHs_python = False
    compute_cov3D_python = False
    debug = False
    antialiasing = False

def render_scene(model_path, poses_csv, output_dir, iterations):
    from scene.cameras import Camera
    os.makedirs(output_dir, exist_ok=True)

    ply_path = f'{model_path}/point_cloud/iteration_{iterations}/point_cloud.ply'
    if not os.path.exists(ply_path):
        print(f'  ❌ PLY not found: {ply_path}')
        return False

    gaussians = GaussianModel(sh_degree=SH_DEGREE)
    gaussians.load_ply(ply_path)

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

        out_name = row['image_name']
        out_ext  = os.path.splitext(out_name)[1].upper()
        out_path = os.path.join(output_dir, out_name)
        if out_ext in ('.JPG', '.JPEG'):
            img_pil.save(out_path, format='JPEG', quality=92, subsampling=0)
        else:
            img_pil.save(out_path, format='PNG', optimize=True, compress_level=6)

        if idx % 15 == 0:
            print(f'    [{idx+1}/{len(df)}] {row["image_name"]}')

    print(f'  ✅ Rendered {len(df)} images → {output_dir}')
    return True

render_log = []
for scene in ALL_SCENES:
    cfg        = SCENE_CONFIG[scene]
    model_path = f'{OUT_DIR}/{scene}'
    poses_csv  = f'{DATA_ROOT}/{scene}/test/test_poses.csv'
    output_dir = f'{RENDER_DIR}/{scene}'

    print(f'\n--- Rendering: {scene} ---')
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

print('\nRENDER SUMMARY:')
for sc, t, s in render_log:
    print(f'  {sc:12s} — {s:8s} — {t:.1f} min')
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
    PSNR_MAX = 40.0
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
        print(f'\n★ Mean Score: {np.mean(all_scores):.4f}')
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
    print('\nErrors:')
    for e in errors: print(f'  {e}')

print('\n=== Creating submission_round2.zip ===')
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
print(f'\n✅ submission_round2.zip: {size_mb:.1f} MB')
if size_mb > 500:
    print('⚠️  WARNING: > 500 MB limit!')
else:
    print('✅ Size OK (< 500 MB)')
print('→ Download từ Kaggle Output panel rồi submit!')