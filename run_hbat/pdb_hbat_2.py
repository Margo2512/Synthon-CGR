import subprocess
from pathlib import Path

INPUT_DIR = Path("cif_files/pdb_output")
OUTPUT_DIR = Path("cif_files/hbat_results")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

pdb_files = list(INPUT_DIR.glob("*.pdb")) + list(INPUT_DIR.glob("*.PDB"))

if not pdb_files:
    print(f"Error: there are no PDB files in {INPUT_DIR}")
    exit(1)

print(f"Found PDB files: {len(pdb_files)}")
print("Run HBAT")

success = 0
failed = 0

for i, pdb_file in enumerate(pdb_files, 1):
    filename = pdb_file.stem
    output_file = OUTPUT_DIR / f"{filename}.txt"
    
    print(f"  [{i}/{len(pdb_files)}] {filename}", end=" ", flush=True)
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            result = subprocess.run(
                ['hbat', str(pdb_file)],
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=30
            )
        
        if result.returncode == 0:
            if output_file.stat().st_size > 100:
                print("OK")
                success += 1
            else:
                print("OK (empty output)")
                success += 1
        else:
            print(f"FAIL (code {result.returncode})")
            failed += 1
            
    except subprocess.TimeoutExpired:
        print("FAIL (Timeout)")
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write("\nERROR: Timeout (300 seconds)\n")
        failed += 1
    except FileNotFoundError:
        print("FAIL (HBAT not found)")
        print("\nError: The 'hbat' command was not found. Make sure that HBAT is installed and available in the PATH")
        exit(1)
    except Exception as e:
        print(f"FAIL ({str(e)[:50]})")
        failed += 1

print(f"\nOk: {success}, Errors: {failed}")
print(f"Results in: {OUTPUT_DIR}")