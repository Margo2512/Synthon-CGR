import re
import csv
from pathlib import Path
from collections import defaultdict

INPUT_DIR = Path("cif_files/hbat_results")
OUTPUT_CSV = Path("cif_files/bonds_ctr_m_final.csv")

def parse_hbonds_from_file(file_path: Path) -> list:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    bonds = []
    
    pattern_hbond_ctr_to_m = r'(\d+)\.\s+H-Bond:\s+([A-Z]):(CTR):(\d+)\(([^)]+)\)\s+-\s+H\s+-\s+([A-Z]):(M\d+):(\d+)\(([^)]+)\)\s+\[([\d.]+)Å,\s+([\d.]+)°\]'
    pattern_hbond_m_to_ctr = r'(\d+)\.\s+H-Bond:\s+([A-Z]):(M\d+):(\d+)\(([^)]+)\)\s+-\s+H\s+-\s+([A-Z]):(CTR):(\d+)\(([^)]+)\)\s+\[([\d.]+)Å,\s+([\d.]+)°\]'
    pattern_xbond = r'(\d+)\.\s+X-Bond:\s+([A-Z]):(M\d+):(\d+)\(([^)]+)\)\s+-\s+([A-Z]):(CTR):(\d+)\(([^)]+)\)\s+\[([\d.]+)Å,\s+([\d.]+)°\]'
    pattern_xbond_reverse = r'(\d+)\.\s+X-Bond:\s+([A-Z]):(CTR):(\d+)\(([^)]+)\)\s+-\s+([A-Z]):(M\d+):(\d+)\(([^)]+)\)\s+\[([\d.]+)Å,\s+([\d.]+)°\]'
    
    for match in re.finditer(pattern_hbond_ctr_to_m, content):
        bond_id = int(match.group(1))
        ctr_atom = match.group(5)
        m_residue = match.group(7)
        m_atom = match.group(9)
        distance = float(match.group(10))
        angle = float(match.group(11))
        
        bond = {
            'file': file_path.stem,
            'type': 'H-bond',
            'bond_id': bond_id,
            'donor': f"CTR:{ctr_atom}",
            'acceptor': f"{m_residue}:{m_atom}",
            'what': f"{ctr_atom}-H···{m_atom}",
            'distance': distance,
            'angle': angle,
        }
        bonds.append(bond)
    
    for match in re.finditer(pattern_hbond_m_to_ctr, content):
        bond_id = int(match.group(1))
        m_residue = match.group(3)
        m_atom = match.group(5)
        ctr_atom = match.group(9)
        distance = float(match.group(10))
        angle = float(match.group(11))
        
        bond = {
            'file': file_path.stem,
            'type': 'H-bond',
            'bond_id': bond_id,
            'donor': f"{m_residue}:{m_atom}",
            'acceptor': f"CTR:{ctr_atom}",
            'what': f"{m_atom}-H···{ctr_atom}",
            'distance': distance,
            'angle': angle,
        }
        bonds.append(bond)
    
    for match in re.finditer(pattern_xbond, content):
        bond_id = int(match.group(1))
        m_chain = match.group(2)
        m_residue = match.group(3)
        m_resnum = int(match.group(4))
        m_atom_full = match.group(5)  
        ctr_chain = match.group(6)
        ctr_residue = match.group(7)
        ctr_resnum = int(match.group(8))
        ctr_atom = match.group(9)      
        distance = float(match.group(10))
        angle = float(match.group(11))
        
        bond = {
            'file': file_path.stem,
            'type': 'X-bond',
            'bond_id': bond_id,
            'donor': f"{m_residue}:{m_atom_full}",
            'acceptor': f"CTR:{ctr_atom}",
            'what': f"{m_atom_full}···{ctr_atom}",  
            'distance': distance,
            'angle': angle,
        }
        bonds.append(bond)
    
    for match in re.finditer(pattern_xbond_reverse, content):
        bond_id = int(match.group(1))
        ctr_chain = match.group(2)
        ctr_residue = match.group(3)
        ctr_resnum = int(match.group(4))
        ctr_atom = match.group(5)
        m_chain = match.group(6)
        m_residue = match.group(7)
        m_resnum = int(match.group(8))
        m_atom_full = match.group(9)
        distance = float(match.group(10))
        angle = float(match.group(11))
        
        bond = {
            'file': file_path.stem,
            'type': 'X-bond',
            'bond_id': bond_id,
            'donor': f"CTR:{ctr_atom}",
            'acceptor': f"{m_residue}:{m_atom_full}",
            'what': f"{ctr_atom}···{m_atom_full}",
            'distance': distance,
            'angle': angle,
        }
        bonds.append(bond)
    
    return bonds

def main():
    txt_files = list(INPUT_DIR.glob("*.txt"))
    
    if not txt_files:
        print(f"Error: there are no TXT files in {INPUT_DIR}")
        return
    
    print(f"Files found: {len(txt_files)}")
    
    all_bonds = []
    
    for txt_file in sorted(txt_files):
        bonds = parse_hbonds_from_file(txt_file)
        if bonds:
            all_bonds.extend(bonds)
            print(f"\n {txt_file.stem}: {len(bonds)} bonds CTR↔M")
            for b in bonds:
                print(f"     #{b['bond_id']} ({b['type']}): {b['donor']} → {b['acceptor']} "
                      f"[{b['distance']:.2f}Å, {b['angle']:.1f}°]")
        else:
            print(f"\n {txt_file.stem}: 0 bonds CTR↔M")
    
    if not all_bonds:
        print("\nCTR ↔ M bonds were not found in any file.")
        return
    
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['File', 'Bond type', 'Bond', 'Donor', 'Acceptor', 'What is formed', 'Distance (Å)', 'Angle (°)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
        writer.writeheader()
        
        for bond in all_bonds:
            row = {
                'File': bond['file'],
                'Bond type': bond['type'],
                'Bond': f"№{bond['bond_id']}",
                'Donor': bond['donor'],
                'Acceptor': bond['acceptor'],
                'What is formed': bond['what'],
                'Distance (Å)': f"{bond['distance']:.2f}",
                'Angle (°)': f"{bond['angle']:.1f}"
            }
            writer.writerow(row)
    
    print("Statistics")
    print(f"Total files: {len(txt_files)}")
    print(f"Files with CTR↔M bonds: {len(set(b['file'] for b in all_bonds))}")
    print(f"Total bonds CTR↔M: {len(all_bonds)}")
    
    hbond_count = sum(1 for b in all_bonds if b['type'] == 'H-bond')
    xbond_count = sum(1 for b in all_bonds if b['type'] == 'X-bond')
    print(f"\nTypes bonds:")
    print(f"  H-bond (hydrogen): {hbond_count}")
    print(f"  X-bond (halogen): {xbond_count}")

    
if __name__ == "__main__":
    main()