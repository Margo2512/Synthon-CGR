import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

INPUT = "data/data.csv"
OUTPUT = "data/data_clean.csv"
REMOVED_OUTPUT = "data/data_removed.csv"

FILTERS = {
    "max_heavy_atoms": 80,
    "max_mol_weight": 1000.0,
}

EXCLUDED_COMPONENTS = {
    "water_hydrate": {
        "water": "O",
    },

    "common_crystallization_solvents": {
        "methanol": "CO",
        "ethanol": "CCO",
        "1-propanol": "CCCO",
        "2-propanol": "CC(O)C",
        "1-butanol": "CCCCO",
        "2-butanol": "CCC(O)C",
        "tert-butanol": "CC(C)(C)O",
        "acetone": "CC(=O)C",
        "acetonitrile": "CC#N",
        "ethyl_acetate": "CCOC(=O)C",
        "diethyl_ether": "CCOCC",
        "tetrahydrofuran": "C1CCOC1",
        "1,4-dioxane": "O1CCOCC1",
        "dichloromethane": "ClCCl",
        "chloroform": "ClC(Cl)Cl",
        "carbon_tetrachloride": "ClC(Cl)(Cl)Cl",
        "dimethylformamide": "CN(C)C=O",
        "dimethyl_sulfoxide": "CS(C)=O",
    },

    "small_inorganic_or_volatile_guests": {
        "ammonia": "N",
        "hydrogen_fluoride": "F",
        "hydrogen_chloride": "Cl",
        "hydrogen_bromide": "Br",
        "hydrogen_iodide": "I",
        "carbon_dioxide": "O=C=O",
        "hydrogen_peroxide": "OO",
        "molecular_nitrogen": "N#N",
        "molecular_oxygen": "O=O",
    },

    "elemental_halogen_guests": {
        "fluorine": "FF",
        "chlorine": "ClCl",
        "bromine": "BrBr",
        "iodine": "II",
    },

    "simple_ions": {
        "fluoride": "[F-]",
        "chloride": "[Cl-]",
        "bromide": "[Br-]",
        "iodide": "[I-]",
        "lithium": "[Li+]",
        "sodium": "[Na+]",
        "potassium": "[K+]",
        "ammonium": "[NH4+]",
        "magnesium": "[Mg+2]",
        "calcium": "[Ca+2]",
    },
}

ENABLED_EXCLUSION_CATEGORIES = {
    "water_hydrate",
    "common_crystallization_solvents",
    "simple_ions",
}


def canonicalize_smiles(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        return None, None

    try:
        mol = Chem.MolFromSmiles(smiles.strip())
    except Exception:
        return None, None

    if mol is None:
        return None, None

    try:
        canonical = Chem.MolToSmiles(
            mol,
            canonical=True,
            isomericSmiles=True,
        )
        return mol, canonical
    except Exception:
        return None, None


def build_exclusion_lookup():
    lookup = {}

    for category in ENABLED_EXCLUSION_CATEGORIES:
        for name, smiles in EXCLUDED_COMPONENTS[category].items():
            mol, canonical = canonicalize_smiles(smiles)

            if mol is None:
                raise ValueError(
                    f"Invalid SMILES in EXCLUDED_COMPONENTS: "
                    f"{category}/{name} -> {smiles}"
                )

            if canonical in lookup:
                previous = lookup[canonical]
                raise ValueError(
                    f"Duplicate exclusion SMILES after canonicalization: "
                    f"{category}/{name} and {previous}"
                )

            lookup[canonical] = (category, name)

    return lookup


EXCLUSION_LOOKUP = build_exclusion_lookup()


def component_status(smiles):
    mol, canonical = canonicalize_smiles(smiles)

    if mol is None:
        return False, None, "invalid_smiles"

    if len(Chem.GetMolFrags(mol)) != 1:
        return False, canonical, "multiple_fragments"

    if canonical in EXCLUSION_LOOKUP:
        category, name = EXCLUSION_LOOKUP[canonical]
        return False, canonical, f"excluded:{category}:{name}"

    net_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
    if net_charge != 0:
        return False, canonical, f"net_formal_charge:{net_charge:+d}"

    if mol.GetNumHeavyAtoms() > FILTERS["max_heavy_atoms"]:
        return False, canonical, "too_many_heavy_atoms"

    if Descriptors.MolWt(mol) > FILTERS["max_mol_weight"]:
        return False, canonical, "molecular_weight_too_high"

    return True, canonical, "ok"


def normalize_pair(smiles1, smiles2):
    if smiles1 <= smiles2:
        return smiles1, smiles2
    return smiles2, smiles1


def clean_dataset(df):
    required = {"SMILES1", "SMILES2", "result"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    work = df.copy()
    print(f"Original dataset size: {len(work)}")

    status1 = work["SMILES1"].apply(component_status)
    status2 = work["SMILES2"].apply(component_status)

    work["_valid1"] = status1.map(lambda x: x[0])
    work["_can1"] = status1.map(lambda x: x[1])
    work["_reason1"] = status1.map(lambda x: x[2])

    work["_valid2"] = status2.map(lambda x: x[0])
    work["_can2"] = status2.map(lambda x: x[1])
    work["_reason2"] = status2.map(lambda x: x[2])

    component_mask = work["_valid1"] & work["_valid2"]

    removed_components = work.loc[~component_mask].copy()
    clean = work.loc[component_mask].copy()

    print(f"After component-level filtering: {len(clean)}")
    print(f"Removed at component level: {len(removed_components)}")

    clean["SMILES1"] = clean["_can1"]
    clean["SMILES2"] = clean["_can2"]

    same_component_mask = clean["SMILES1"] == clean["SMILES2"]
    removed_same = clean.loc[same_component_mask].copy()
    clean = clean.loc[~same_component_mask].copy()

    print(f"Removed A == B pairs: {len(removed_same)}")

    normalized = clean.apply(
        lambda row: normalize_pair(row["SMILES1"], row["SMILES2"]),
        axis=1,
    )

    clean["SMILES1"] = normalized.map(lambda x: x[0])
    clean["SMILES2"] = normalized.map(lambda x: x[1])
    clean["_pair_key"] = clean["SMILES1"] + "||" + clean["SMILES2"]

    label_counts = clean.groupby("_pair_key")["result"].nunique(dropna=False)
    conflicting_keys = set(label_counts[label_counts > 1].index)

    conflict_mask = clean["_pair_key"].isin(conflicting_keys)
    removed_conflicts = clean.loc[conflict_mask].copy()
    clean = clean.loc[~conflict_mask].copy()

    print(f"Conflicting pair keys: {len(conflicting_keys)}")
    print(f"Rows removed because of label conflicts: {len(removed_conflicts)}")

    n_before = len(clean)

    clean = clean.drop_duplicates(
        subset=["_pair_key", "result"],
        keep="first",
    ).copy()

    print(f"Duplicate pairs removed: {n_before - len(clean)}")

    removed_components["_removal_stage"] = "component_filter"
    removed_same["_removal_stage"] = "same_component"
    removed_conflicts["_removal_stage"] = "label_conflict"

    removed = pd.concat(
        [removed_components, removed_same, removed_conflicts],
        ignore_index=True,
        sort=False,
    )

    helper_cols = [
        "_valid1", "_can1", "_reason1",
        "_valid2", "_can2", "_reason2",
        "_pair_key",
    ]

    clean = clean.drop(
        columns=[c for c in helper_cols if c in clean.columns],
        errors="ignore",
    ).reset_index(drop=True)

    print(f"\nFinal dataset size: {len(clean)}")
    print("Class distribution:")
    print(clean["result"].value_counts(dropna=False).sort_index())

    if len(removed_components):
        print("\nRemoval reasons for SMILES1:")
        print(
            removed_components["_reason1"]
            .value_counts(dropna=False)
            .head(30)
        )

        print("\nRemoval reasons for SMILES2:")
        print(
            removed_components["_reason2"]
            .value_counts(dropna=False)
            .head(30)
        )

    return clean, removed


def main():
    df = pd.read_csv(INPUT)

    clean, removed = clean_dataset(df)

    clean.to_csv(OUTPUT, index=False)
    removed.to_csv(REMOVED_OUTPUT, index=False)

    print(f"\nSaved clean dataset: {OUTPUT}")
    print(f"Saved removed-row audit: {REMOVED_OUTPUT}")


if __name__ == "__main__":
    main()
