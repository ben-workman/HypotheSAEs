"""sae_feature_selection.py (local script)

Teacher VA example using the *single-dictionary* SAE workflow (no meta-features).

This file is intentionally kept OUTSIDE the installable `hypothesaes/` package so it
doesn't ship to PyPI. It's a practical, runnable script version of the original
Colab notebook, with:
- the same data-loading/merging logic
- no hardcoded API keys
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

from HypotheSAEs.hypothesaes.quickstart import set_seed, train_sae, generate_hypotheses


def load_teacher_va_data_from_drive(
    *,
    drive_root: str = "/content/drive/MyDrive/Research/Predicting Teacher Value-Added",
    embeddings_csv: str = "Data/Constructed Data/Embeddings/OpenAI Embeddings/openai_embeddings_chunked_smaller.csv",
    transcripts_csv: str = "Data/Constructed Data/cleaned_ncte_transcripts_chunked_smaller.csv",
    teacher_info_csv: str = "Data/Constructed Data/9_12_24/teacher_characteristics.csv",
    train_ids_csv: str = "Data/Constructed Data/9_12_24/train_nctetid.csv",
    train_id_col: str = "NCTETID",
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Load the same data as the original Colab file.

    Returns:
        (train_texts, train_embeddings, train_labels, train_group_ids,
         test_texts, test_embeddings, test_labels, test_group_ids)
    """
    embeddings = pd.read_csv(os.path.join(drive_root, embeddings_csv))
    transcripts = pd.read_csv(os.path.join(drive_root, transcripts_csv))
    teacher_info = pd.read_csv(os.path.join(drive_root, teacher_info_csv))

    df = pd.DataFrame(
        {
            "observation_id": embeddings.iloc[:, 0].astype(str).values,
            "NCTETID": embeddings.iloc[:, 1].astype(int).values,
            "year_suffix": embeddings.iloc[:, 2].astype(int).values,
        }
    )
    df["embedding"] = list(embeddings.iloc[:, 3:].values.tolist())

    transcripts = transcripts.copy()
    transcripts["observation_id"] = transcripts["observation_id"].astype(str)

    data = (
        df.merge(transcripts[["observation_id", "full_text"]], on="observation_id", how="left")
        .merge(teacher_info[["NCTETID", "STATEVA_M"]], on="NCTETID", how="left", validate="m:1")
        .rename(columns={"STATEVA_M": "label"})
        .dropna(subset=["label"])
    )

    uids = data["NCTETID"].to_numpy()
    train_df = pd.read_csv(os.path.join(drive_root, train_ids_csv))

    # split by teacher IDs
    uids_str = pd.Series(uids, dtype=str).str.strip().to_numpy()
    train_ids_str = train_df[train_id_col].astype(str).str.strip().to_numpy()
    mask = np.isin(uids_str, train_ids_str)

    train_ids_all = uids[mask]
    test_ids = uids[~mask]

    train_data_all = data[data["NCTETID"].isin(train_ids_all)]
    test_data = data[data["NCTETID"].isin(test_ids)]

    train_texts = train_data_all["full_text"].tolist()
    train_labels = train_data_all["label"].to_numpy()
    train_embeddings = np.stack(train_data_all["embedding"].to_list())
    train_group_ids = train_data_all["NCTETID"].to_numpy()

    test_texts = test_data["full_text"].tolist()
    test_labels = test_data["label"].to_numpy()
    test_embeddings = np.stack(test_data["embedding"].to_list())
    test_group_ids = test_data["NCTETID"].to_numpy()

    return (
        train_texts,
        train_embeddings,
        train_labels,
        train_group_ids,
        test_texts,
        test_embeddings,
        test_labels,
        test_group_ids,
    )

# ============================================================================
# Notebook-style execution (ready to run in Colab)
# ============================================================================
# Setup (run these in Colab):
# !git clone https://github.com/ben-workman/HypotheSAEs
# !pip install hypothesaes
#
# Do NOT hardcode keys; set them in Colab via Secrets or:
# os.environ["OPENAI_API_KEY"] = "..."
# os.environ["OPENAI_KEY_SAE"] = "..."

try:
    from google.colab import drive  # type: ignore

    drive.mount("/content/drive")
except Exception:
    # Non-Colab environment
    pass

set_seed(123)

# 1) Load data (same logic as original notebook)
(
    train_texts,
    train_embeddings,
    train_labels,
    train_group_ids,
    _test_texts,
    _test_embeddings,
    _test_labels,
    _test_group_ids,
) = load_teacher_va_data_from_drive()

print("Train texts:", len(train_texts))
print("Train embeddings shape:", train_embeddings.shape)
print("Unique teachers (train):", len(np.unique(train_group_ids)))

# 2) Train a single SAE (matryoshka; keep small for sanity checks)
checkpoint_dir = "/content/drive/MyDrive/Research/Predicting Teacher Value-Added/SAEs/model_checkpoint/"

sae = train_sae(
    embeddings=train_embeddings,
    M=[64, 128, 256],
    K=32,
    aux_coef=1 / 4,
    checkpoint_dir=checkpoint_dir,
    n_epochs=50,
    patience=3,
    batch_size=1024,
    overwrite_checkpoint=True,
)

# 3) Generate hypotheses (existing selection + interpretation infra)
hypotheses_df = generate_hypotheses(
    texts=train_texts,
    labels=train_labels,
    embeddings=train_embeddings,
    sae=sae,
    cache_name="teacher_va_single_sae",
    group_ids=train_group_ids,
    selection_method="stability",
    n_selected_neurons=20,
    n_candidate_interpretations=2,
    n_scoring_examples=60,
    filter=False,
)

out_path = "/content/drive/MyDrive/Research/Predicting Teacher Value-Added/Data/Constructed Data/9_12_24/feature_information_single_sae.csv"
hypotheses_df.to_csv(out_path, index=False)
print("Saved:", out_path)


