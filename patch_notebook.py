"""
patch_notebook.py — Sửa nvs_pipeline.ipynb để giảm kích thước submission từ ~1.6GB → ~220MB

Vấn đề:
  - Notebook đang đổi tất cả tên file thành .png và lưu PNG
  - 724 ảnh PNG 1320x989 ≈ 1.1-1.6 GB
  - Nhưng test_poses.csv yêu cầu tên file .JPG (DJI drone photos)

Fix:
  1. Cell 5: Giữ đúng tên file gốc từ CSV (image_name), lưu JPEG quality=92
  2. Cell 7: Đổi tên ZIP → submission_round1.zip, tìm cả .JPG/.jpg/.png,
             dùng ZIP_STORED (JPEG đã nén rồi, DEFLATE không giúp nhiều)
"""

import json, re, sys, os

NB_PATH = os.path.join(os.path.dirname(__file__), 'nvs_pipeline.ipynb')

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

def join_source(cell):
    return ''.join(cell['source'])

def set_source(cell, new_source_str):
    # Split into list of lines, keeping \n at end of each line
    lines = new_source_str.splitlines(keepends=True)
    # Last line should NOT have \n (Jupyter convention)
    if lines and lines[-1].endswith('\n'):
        lines[-1] = lines[-1].rstrip('\n')
    cell['source'] = lines

patched = {'cell5': False, 'cell7': False}

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    src = join_source(cell)

    # ---- CELL 5 FIX: Image saving (PNG → JPEG) ----
    if "CELL 5" in src and "out_name = os.path.splitext(row['image_name'])[0] + '.png'" in src:
        old = "        # Đặt tên file output giống image_name nhưng đổi extension → .png\n        out_name = os.path.splitext(row['image_name'])[0] + '.png'\n        img_pil.save(os.path.join(output_dir, out_name))"
        new = (
            "        # Giữ đúng tên file gốc từ image_name (thường là .JPG từ DJI drone)\n"
            "        out_name = row['image_name']  # Giữ nguyên tên + extension từ CSV\n"
            "        out_ext  = os.path.splitext(out_name)[1].upper()\n"
            "        out_path = os.path.join(output_dir, out_name)\n"
            "        if out_ext in ('.JPG', '.JPEG'):\n"
            "            img_pil.save(out_path, format='JPEG', quality=92, subsampling=0)\n"
            "        else:\n"
            "            img_pil.save(out_path, format='PNG', optimize=True, compress_level=9)"
        )
        if old in src:
            src = src.replace(old, new)
            set_source(cell, src)
            patched['cell5'] = True
            print("✅ Cell 5 patched: PNG → JPEG saving")
        else:
            print("⚠️  Cell 5: exact pattern not found, trying regex...")
            # Fallback regex
            pattern = r"# Đặt tên file output.*?img_pil\.save\(os\.path\.join\(output_dir, out_name\)\)"
            if re.search(pattern, src, re.DOTALL):
                src = re.sub(pattern, new, src, flags=re.DOTALL)
                set_source(cell, src)
                patched['cell5'] = True
                print("✅ Cell 5 patched via regex")
            else:
                print("❌ Cell 5: could not find save pattern!")

    # ---- CELL 7 FIX: Submission ZIP ----
    if "CELL 7" in src and "submission.zip" in src:
        # Fix 1: Rename zip file
        src = src.replace(
            "ZIP_PATH = '/kaggle/working/submission.zip'",
            "ZIP_PATH = '/kaggle/working/submission_round1.zip'"
        )

        # Fix 2: Replace the entire zipfile creation block
        old_zip_block = (
            "print('\\n=== Creating submission.zip ===')\n"
            "with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zf:\n"
            "    for dataset, scene in ALL_SCENES:\n"
            "        render_dir = f'{RENDER_DIR}/{scene}'\n"
            "        imgs = sorted(\n"
            "            glob.glob(f'{render_dir}/*.png') +\n"
            "            glob.glob(f'{render_dir}/*.jpg')\n"
            "        )\n"
            "        for img_path in imgs:\n"
            "            arcname = f'{scene}/{os.path.basename(img_path)}'\n"
            "            zf.write(img_path, arcname)\n"
            "        print(f'  {scene}: {len(imgs)} images')\n"
            "\n"
            "size_mb = os.path.getsize(ZIP_PATH) / 1e6\n"
            "print(f'\\n✅ submission.zip created: {size_mb:.1f} MB')\n"
            "print(f'   Path: {ZIP_PATH}')\n"
            "print('   → Download từ Kaggle Output panel rồi submit!')"
        )
        new_zip_block = (
            "print('\\n=== Creating submission_round1.zip ===')\n"
            "# Use ZIP_STORED for JPEG (already compressed) — DEFLATE saves <1%\n"
            "with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_STORED) as zf:\n"
            "    for dataset, scene in ALL_SCENES:\n"
            "        render_dir = f'{RENDER_DIR}/{scene}'\n"
            "        if not os.path.exists(render_dir):\n"
            "            print(f'  ⚠️ {scene}: render dir not found, skipping')\n"
            "            continue\n"
            "        imgs = sorted(\n"
            "            glob.glob(f'{render_dir}/*.png') +\n"
            "            glob.glob(f'{render_dir}/*.PNG') +\n"
            "            glob.glob(f'{render_dir}/*.jpg') +\n"
            "            glob.glob(f'{render_dir}/*.JPG')\n"
            "        )\n"
            "        for img_path in imgs:\n"
            "            arcname = f'{scene}/{os.path.basename(img_path)}'\n"
            "            zf.write(img_path, arcname)\n"
            "        print(f'  {scene}: {len(imgs)} images')\n"
            "\n"
            "size_mb = os.path.getsize(ZIP_PATH) / 1e6\n"
            "print(f'\\n✅ submission_round1.zip created: {size_mb:.1f} MB')\n"
            "print(f'   Path: {ZIP_PATH}')\n"
            "if size_mb > 500:\n"
            "    print(f'   ⚠️  WARNING: {size_mb:.1f} MB > 500 MB limit! Consider lowering JPEG quality.')\n"
            "else:\n"
            "    print(f'   ✅ Size OK ({size_mb:.1f} MB < 500 MB limit)')\n"
            "print('   → Download từ Kaggle Output panel rồi submit!')"
        )

        # Fix 3: Better validation (check by filename not just count)
        old_validate = (
            "    found = len(glob.glob(f'{render_dir}/*.png') + glob.glob(f'{render_dir}/*.jpg'))\n"
            "    status = '✅' if found == expected else f'⚠️  {found}/{expected}'\n"
            "    print(f'  {scene:15s}: {status}')\n"
            "    if found != expected:\n"
            "        errors.append(f'{scene}: expected {expected}, got {found}')"
        )
        new_validate = (
            "    expected_names = set(df['image_name'].tolist())\n"
            "    found_files = set(os.path.basename(p) for p in\n"
            "        glob.glob(f'{render_dir}/*.png') +\n"
            "        glob.glob(f'{render_dir}/*.PNG') +\n"
            "        glob.glob(f'{render_dir}/*.jpg') +\n"
            "        glob.glob(f'{render_dir}/*.JPG'))\n"
            "    found = len(found_files)\n"
            "    missing = expected_names - found_files\n"
            "    status = '✅' if not missing else f'⚠️  {found}/{expected}'\n"
            "    print(f'  {scene:15s}: {status}')\n"
            "    if missing:\n"
            "        sample = list(missing)[:5]\n"
            "        print(f'    Missing ({len(missing)}): {sample}' + ('...' if len(missing)>5 else ''))\n"
            "        errors.append(f'{scene}: missing {len(missing)} images')"
        )

        changed = False
        if old_zip_block in src:
            src = src.replace(old_zip_block, new_zip_block)
            changed = True
        else:
            print("⚠️  Cell 7: zip block pattern not found exactly — skipping zip block patch")

        if old_validate in src:
            src = src.replace(old_validate, new_validate)
        else:
            print("⚠️  Cell 7: validate block pattern not found — skipping validate patch")

        set_source(cell, src)
        patched['cell7'] = True
        print("✅ Cell 7 patched: submission_round1.zip + JPEG support + validation improved")

print("\n=== Patch Summary ===")
for k, v in patched.items():
    status = "✅ Applied" if v else "❌ NOT Applied"
    print(f"  {k}: {status}")

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"\n✅ Saved patched notebook → {NB_PATH}")
print("\n📌 KEY CHANGES:")
print("  Cell 5: Lưu ảnh theo đúng tên từ CSV (giữ .JPG), JPEG quality=92")
print("         → giảm từ ~3MB/ảnh (PNG) xuống ~300KB/ảnh (JPEG)")
print("  Cell 7: ZIP tên submission_round1.zip, dùng ZIP_STORED (JPEG không cần nén lại)")
print("         → tìm .JPG/.jpg/.png/.PNG files")
print(f"\n📊 Ước tính kích thước submission_round1.zip:")
print(f"  724 ảnh × ~280KB/ảnh (JPEG q92) ≈ ~200 MB  ✅ (giới hạn 500 MB)")
