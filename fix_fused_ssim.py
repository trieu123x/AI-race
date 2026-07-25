"""
Fix fused-ssim installation in nvs_pipeline_round2.ipynb:
  - fused-ssim KHÔNG có trên PyPI
  - Phải cài từ submodule đã clone: GS_DIR/submodules/fused-ssim
"""
import json, re, sys

NB_PATH = r"c:\Users\admin\Downloads\VAI_NVS_DATA\nvs_pipeline_round2.ipynb"

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

OLD_COMMENT = "# fused-ssim: SSIM mem-efficient (tiet kiem ~400MB VRAM/call, khong doi chat luong)\n"
NEW_COMMENT = "# fused-ssim: khong co tren PyPI, phai cai tu submodule da clone\n"

OLD_CMD = "    subprocess.run(['pip','install','-q','fused-ssim'], check=True)\n"
NEW_CMD = "    subprocess.run(['pip','install','-q',f'{GS_DIR}/submodules/fused-ssim'], check=True)\n"

patched = False
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = cell["source"]
    new_src = []
    for line in src:
        if line == OLD_COMMENT:
            new_src.append(NEW_COMMENT)
            patched = True
        elif line == OLD_CMD:
            new_src.append(NEW_CMD)
        else:
            new_src.append(line)
    cell["source"] = new_src

if patched:
    with open(NB_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("PATCHED: fused-ssim now installs from local submodule path.")
else:
    print("NOT FOUND -- may already be patched or pattern mismatch.")
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code" and any("fused-ssim" in l for l in cell["source"]):
            print(f"\nCell {i} relevant lines:")
            for l in cell["source"]:
                if "fused" in l or "subprocess" in l:
                    print(repr(l))
