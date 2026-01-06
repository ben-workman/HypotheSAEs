"""meta_sae_feature_selection.py (local script)

Teacher VA example using the *meta-feature* SAE workflow:
- Train an ensemble of SAEs with different seeds
- Align + cluster features across runs into meta-features
- Compute pooled meta-activations
- Stability select meta-features predictive of y (supports per-sample or per-group y)
- Interpret member features, then synthesize a single meta-feature description
- Score the synthesized interpretation (fidelity) 

This file is intentionally kept OUTSIDE the installable `hypothesaes/` package.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

from HypotheSAEs.hypothesaes.quickstart import set_seed
from HypotheSAEs.hypothesaes.meta_features import build_meta_features, interpret_meta_features
from HypotheSAEs.hypothesaes.select_neurons import select_neurons_stability
from HypotheSAEs.hypothesaes.interpret_neurons import NeuronInterpreter, ScoringConfig, sample_top_zero


def load_teacher_va_data_from_drive(
    *,
    drive_root: str = "/content/drive/MyDrive/Research/Predicting Teacher Value-Added",
    embeddings_csv: str = "Data/Constructed Data/Embeddings/OpenAI Embeddings/openai_embeddings_chunked_smaller.csv",
    transcripts_csv: str = "Data/Constructed Data/cleaned_ncte_transcripts_chunked_smaller.csv",
    teacher_info_csv: str = "Data/Constructed Data/9_12_24/teacher_characteristics.csv",
    train_ids_csv: str = "Data/Constructed Data/9_12_24/train_nctetid.csv",
    train_id_col: str = "NCTETID",
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Load the same data as the original Colab file (train split only)."""
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

    uids_str = pd.Series(uids, dtype=str).str.strip().to_numpy()
    train_ids_str = train_df[train_id_col].astype(str).str.strip().to_numpy()
    mask = np.isin(uids_str, train_ids_str)

    train_ids_all = uids[mask]
    train_data_all = data[data["NCTETID"].isin(train_ids_all)]

    train_texts = train_data_all["full_text"].tolist()
    train_labels = train_data_all["label"].to_numpy()
    train_embeddings = np.stack(train_data_all["embedding"].to_list())
    train_group_ids = train_data_all["NCTETID"].to_numpy()

    return train_texts, train_embeddings, train_labels, train_group_ids


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
train_texts, train_embeddings, train_labels, train_group_ids = load_teacher_va_data_from_drive()
print("Train texts:", len(train_texts))
print("Train embeddings shape:", train_embeddings.shape)
print("Unique teachers (train):", len(np.unique(train_group_ids)))

# 2) Train SAE ensemble + build meta-features
# Start small so we validate correctness before spending compute.
seeds = [10, 20, 30]
M = [64, 128, 256]
K = 32

# Save checkpoints (and any prototype caches) under Drive
checkpoint_dir = "/content/drive/MyDrive/Research/Predicting Teacher Value-Added/SAEs/meta_sae_checkpoints/"

meta_feature_set = build_meta_features(
    embeddings=train_embeddings,
    M=M,
    K=K,
    seeds=seeds,
    checkpoint_dir=checkpoint_dir,
    # Similarity/clustering
    alpha=0.5,
    cluster_threshold=0.7,
    min_support=0.5,  # keep only meta-features appearing in >=50% of runs
    # Training args (tune later)
    n_epochs=50,
    patience=3,
    batch_size=1024,
    aux_coef=1 / 4,
    overwrite_checkpoint=True,
)

print("Meta-features:", meta_feature_set.n_meta_features)

# 3) Build meta-activations and run stability selection
meta_acts = meta_feature_set.get_activations(
    X=train_embeddings,
    pooling="softmax",
    tau=1.0,
    standardize=True,
)

selected, selected_pi = select_neurons_stability(
    activations=meta_acts,
    target=train_labels,
    group_ids=train_group_ids,  # enables group-level subsampling
    group_subsample=True,
    n_bootstrap=100,
    q=20,
    pi_threshold=0.6,
    random_state=123,
    return_diagnostics=False,
)

if not selected:
    raise RuntimeError("No meta-features selected. Try lowering pi_threshold or min_support.")

# Limit how many we interpret for the smoke test
selected = selected[:5]
selected_pi = selected_pi[:5]

# 4) Interpret meta-features (member interps -> synthesized interp)
meta_interp = interpret_meta_features(
    texts=train_texts,
    embeddings=train_embeddings,
    meta_feature_set=meta_feature_set,
    meta_feature_indices=selected,
    n_members_per_cluster=2,
    n_candidates_per_member=2,
    interpreter_model="gpt-4o",
    synthesizer_model="gpt-4o-mini",
    annotator_model="gpt-4o-mini",
    cache_name="teacher_va_meta_sae",
    n_workers=6,
    show_progress=True,
)

# Score synthesized interpretations against the *meta-activation* signal
interpreter = NeuronInterpreter(
    interpreter_model="gpt-4o",
    annotator_model="gpt-4o-mini",
    cache_name="teacher_va_meta_sae",
)
scoring_cfg = ScoringConfig(n_examples=60, sampling_function=sample_top_zero)

synth_interps = {i: [meta_interp[mf]["synthesized_interpretation"]] for i, mf in enumerate(selected)}
score_acts = meta_acts[:, selected]
metrics = interpreter.score_interpretations(
    texts=train_texts,
    activations=score_acts,
    interpretations=synth_interps,
    config=scoring_cfg,
)

# 5) Save results (includes frequency across runs via support / n_runs)
rows = []
for i, (mf_idx, pi) in enumerate(zip(selected, selected_pi)):
    info = meta_interp[mf_idx]
    synth = info["synthesized_interpretation"]
    f1 = metrics.get(i, {}).get(synth, {}).get("f1")
    rows.append(
        {
            "meta_feature_idx": int(mf_idx),
            "selection_pi": float(pi),
            # Representation stability
            "support": float(info["support"]),  # fraction of runs
            "n_runs_in_cluster": int(info["n_runs"]),  # count of runs
            "coherence": float(info["coherence"]),
            "n_members": int(info["n_members"]),
            # Interpretation
            "synthesized_interpretation": synth,
            "f1_fidelity_score": None if f1 is None else float(f1),
        }
    )

results_df = pd.DataFrame(rows).sort_values("selection_pi", ascending=False).reset_index(drop=True)

out_path = "/content/drive/MyDrive/Research/Predicting Teacher Value-Added/Data/Constructed Data/9_12_24/meta_feature_information.csv"
results_df.to_csv(out_path, index=False)
print("Saved:", out_path)


