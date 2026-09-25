import subprocess
from pathlib import Path


SCRIPT_PATH = Path("cif_to_pdb_hbat.py")
INPUT_DIR = Path("cif_files/cif_files_all_without_duplicates")
OUTPUT_DIR = Path("cif_files/pdb_output")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

cif_files = list(INPUT_DIR.glob("*.cif")) + list(INPUT_DIR.glob("*.CIF"))

print(f"Found CIF files: {len(cif_files)}")

success = 0
for i, cif_file in enumerate(cif_files, 1):
    filename = cif_file.stem
    output_file = OUTPUT_DIR / f"{filename}.pdb"
    
    print(f"  [{i}/{len(cif_files)}] {filename}", end=" ", flush=True)
    
    result = subprocess.run([
        "python3", str(SCRIPT_PATH),
        str(cif_file),
        "-o", str(output_file)
    ], capture_output=True)
    
    if result.returncode == 0 and output_file.exists():
        success += 1
    else:
        print(result.stderr)

print(f"\nOk: {success}/{len(cif_files)}")