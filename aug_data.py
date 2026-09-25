import random
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

warnings.filterwarnings("ignore")


Pair = Tuple[str, str]
NegativeSample = Tuple[str, str, int]


def smiles_to_ecfp4(
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048,
) -> Optional[np.ndarray]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits,
    )
    return np.asarray(generator.GetFingerprint(mol), dtype=np.uint8)


def tanimoto_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
    intersection = np.sum(fp1 & fp2)
    union = np.sum(fp1 | fp2)
    return float(intersection / union) if union > 0 else 0.0


def canonical_pair(smiles1: str, smiles2: str) -> Pair:
    return tuple(sorted((smiles1, smiles2)))


def load_and_preprocess_data(csv_file: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file).copy()

    required = {"SMILES1", "SMILES2", "result"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    all_smiles = pd.unique(df[["SMILES1", "SMILES2"]].values.ravel())
    smiles_to_fp = {
        smiles: smiles_to_ecfp4(smiles)
        for smiles in all_smiles
        if isinstance(smiles, str)
    }

    df["fp1"] = df["SMILES1"].map(smiles_to_fp)
    df["fp2"] = df["SMILES2"].map(smiles_to_fp)
    df = df.dropna(subset=["SMILES1", "SMILES2", "fp1", "fp2"]).reset_index(drop=True)

    return df


def build_positive_coformer_map(positive_df: pd.DataFrame) -> Dict[str, Set[str]]:
    coformers: Dict[str, Set[str]] = defaultdict(set)

    for _, row in positive_df.iterrows():
        a = row["SMILES1"]
        b = row["SMILES2"]
        coformers[a].add(b)
        coformers[b].add(a)

    return coformers


def generate_negative_samples(
    df: pd.DataFrame,
    similarity_threshold: float = 0.6,
    target_count: Optional[int] = None,
    max_attempts_per_source: int = 100,
    random_state: int = 42,
) -> List[NegativeSample]:
    positive_df = df[df["result"] == 1].reset_index(drop=True)
    n_positive = len(positive_df)
    n_existing_negative = int((df["result"] == 0).sum())

    if positive_df.empty:
        print("No positive co-crystal pairs were found.")
        return []

    if n_positive < 2:
        raise ValueError(
            "At least two positive co-crystal pairs are required for augmentation."
        )

    if target_count is None:
        target_count = max(0, n_positive - n_existing_negative)

    if target_count == 0:
        print(
            "The dataset is already balanced or has at least as many "
            "negatives as positives."
        )
        return []

    smiles_to_fp: Dict[str, np.ndarray] = {}
    for _, row in positive_df.iterrows():
        smiles_to_fp[row["SMILES1"]] = row["fp1"]
        smiles_to_fp[row["SMILES2"]] = row["fp2"]

    coformer_map = build_positive_coformer_map(positive_df)

    existing_pairs = {
        canonical_pair(row["SMILES1"], row["SMILES2"])
        for _, row in df.iterrows()
    }

    generated_pairs: Set[Pair] = set()
    negative_samples: List[NegativeSample] = []
    rng = random.Random(random_state)

    diagnostics = {
        "self_pair": 0,
        "existing_pair": 0,
        "duplicate_generated": 0,
        "similar_to_known_coformer": 0,
        "source_pairs_without_negative": 0,
    }

    source_indices = list(range(n_positive))
    rng.shuffle(source_indices)

    print(
        f"Target synthetic negatives: {target_count} "
        f"(positive={n_positive}, existing negative={n_existing_negative})"
    )
    print(
        f"Single pass: each positive source pair is visited once, "
        f"with up to {max_attempts_per_source} candidate attempts."
    )

    for source_idx in source_indices:
        if len(negative_samples) >= target_count:
            break

        source_row = positive_df.iloc[source_idx]

        fixed = (
            source_row["SMILES1"]
            if rng.random() < 0.5
            else source_row["SMILES2"]
        )

        found = False

        for _ in range(max_attempts_per_source):
            other_idx = rng.randrange(n_positive - 1)
            if other_idx >= source_idx:
                other_idx += 1

            other_row = positive_df.iloc[other_idx]

            candidate = (
                other_row["SMILES1"]
                if rng.random() < 0.5
                else other_row["SMILES2"]
            )

            if candidate == fixed:
                diagnostics["self_pair"] += 1
                continue

            new_pair = canonical_pair(fixed, candidate)

            if new_pair in existing_pairs:
                diagnostics["existing_pair"] += 1
                continue

            if new_pair in generated_pairs:
                diagnostics["duplicate_generated"] += 1
                continue

            candidate_fp = smiles_to_fp.get(candidate)
            if candidate_fp is None:
                continue

            reject = False
            for known_coformer in coformer_map.get(fixed, set()):
                known_fp = smiles_to_fp.get(known_coformer)
                if known_fp is None:
                    continue

                tc = tanimoto_similarity(candidate_fp, known_fp)

                if tc >= similarity_threshold:
                    diagnostics["similar_to_known_coformer"] += 1
                    reject = True
                    break

            if reject:
                continue

            generated_pairs.add(new_pair)
            negative_samples.append((fixed, candidate, 0))
            found = True
            break

        if not found:
            diagnostics["source_pairs_without_negative"] += 1

    print("\nAugmentation summary")
    print(f"  Positive pairs: {n_positive}")
    print(f"  Existing negative pairs: {n_existing_negative}")
    print(f"  Target synthetic negatives: {target_count}")
    print(f"  Synthetic negatives generated: {len(negative_samples)}")
    print(
        f"  Final class counts: positive={n_positive}, "
        f"negative={n_existing_negative + len(negative_samples)}"
    )

    if len(negative_samples) < target_count:
        print(
            "\nTarget was not reached in the single pass. "
            "The algorithm stops here as specified."
        )
        print(
            f"Missing synthetic negatives: "
            f"{target_count - len(negative_samples)}"
        )

    print("  Rejection diagnostics:")
    for key, value in diagnostics.items():
        print(f"    {key}: {value}")

    return negative_samples


def main() -> None:
    input_file = (
        "data/cocrystal_data.csv"
    )
    output_file = (
        "data/cocrystal_data_balanced.csv"
    )

    df = load_and_preprocess_data(input_file)

    print(f"Loaded rows: {len(df)}")
    print(f"Positive: {len(df[df['result'] == 1])}")
    print(f"Negative: {len(df[df['result'] == 0])}")

    n_positive = int((df["result"] == 1).sum())
    n_existing_negative = int((df["result"] == 0).sum())
    target_synthetic_negatives = max(0, n_positive - n_existing_negative)

    negative_samples = generate_negative_samples(
        df,
        similarity_threshold=0.6,
        target_count=target_synthetic_negatives,
        max_attempts_per_source=100,
        random_state=42,
    )

    negative_df = pd.DataFrame(
        negative_samples,
        columns=["SMILES1", "SMILES2", "result"],
    )

    augmented_df = pd.concat(
        [df[["SMILES1", "SMILES2", "result"]], negative_df],
        ignore_index=True,
    )
    augmented_df = augmented_df.sample(frac=1, random_state=42).reset_index(drop=True)
    augmented_df.to_csv(output_file, index=False)

    print(f"Final dataset: {len(augmented_df)} rows")
    print(f"Positive: {len(augmented_df[augmented_df['result'] == 1])}")
    print(f"Negative: {len(augmented_df[augmented_df['result'] == 0])}")


if __name__ == "__main__":
    main()
