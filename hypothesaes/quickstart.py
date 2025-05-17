"""High-level functions for hypothesis generation using SAEs."""

import numpy as np
import pandas as pd
from typing import List, Optional, Union, Tuple, Dict, Callable, Any
import torch
import os, openai 
from pathlib import Path
import random

from .sae import SparseAutoencoder, load_model
from .select_neurons import select_neurons
from .interpret_neurons import NeuronInterpreter, InterpretConfig, ScoringConfig, LLMConfig, SamplingConfig, sample_top_zero, sample_percentile_bins
from .utils import get_text_for_printing
from .annotate import annotate_texts_with_concepts
from .evaluation import score_hypotheses
BASE_DIR = Path(__file__).parent.parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

def set_seed(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

def train_sae(
    embeddings: Union[list, np.ndarray],
    M: Union[int, list],  
    K: int,  
    *,
    checkpoint_dir: Optional[str] = None,
    overwrite_checkpoint: bool = False,
    val_embeddings: Optional[Union[list, np.ndarray]] = None,
    aux_k: Optional[int] = None,
    multi_k: Optional[int] = None,
    dead_neuron_threshold_steps: int = 256,
    batch_size: int = 512,
    learning_rate: float = 5e-4,
    n_epochs: int = 100,
    aux_coef: float = 1/32,
    multi_coef: float = 0.0,
    patience: int = 3,
    clip_grad: float = 1.0,
) -> SparseAutoencoder:
    """Train a Sparse Autoencoder or load an existing one.
    
    Args:
        embeddings: Pre-computed embeddings for training (list or numpy array).
        M: Number of neurons in SAE. If provided as a list [m1, m2, ..., mn] (with m1 < ... < mn),
           the model trains a Matryoshka Sparse Autoencoder with nested dictionary sizes.
        K: Number of top-activating neurons to keep per forward pass. If provided as a list, it specifies
           the corresponding active neuron counts for each nested sub-SAE.
        checkpoint_dir: Optional directory for storing/loading SAE checkpoints.
        val_embeddings: Optional validation embeddings for early stopping during SAE training.
        aux_k: Number of neurons to consider for dead neuron revival.
        multi_k: Number of neurons for secondary reconstruction.
        dead_neuron_threshold_steps: Number of non-firing steps after which a neuron is considered dead.
        batch_size: Batch size for training.
        learning_rate: Learning rate for training.
        n_epochs: Maximum number of training epochs.
        aux_coef: Coefficient for auxiliary loss.
        multi_coef: Coefficient for multi-k loss.
        patience: Early stopping patience.
        clip_grad: Gradient clipping value.
        
    Returns:
        Trained SparseAutoencoder model.
    """

    embeddings = np.array(embeddings)
    input_dim = embeddings.shape[1]
    
    X = torch.tensor(embeddings, dtype=torch.float32).to(device)
    X_val = torch.tensor(val_embeddings, dtype=torch.float32).to(device) if val_embeddings is not None else None
    
    def _format_param_for_filename(param):
        if isinstance(param, (list, tuple, np.ndarray)):
            return "-".join(str(int(x)) for x in param)
        else:
            return str(int(param)) 

    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        m_str = _format_param_for_filename(M)
        k_str = _format_param_for_filename(K)
        checkpoint_path = os.path.join(checkpoint_dir, f"SAE_M={m_str}_K={k_str}.pt")
        if os.path.exists(checkpoint_path) and not overwrite_checkpoint:
            return load_model(checkpoint_path).to(device)
    
    sae = SparseAutoencoder(
        input_dim=input_dim,
        m_total_neurons=M,
        k_active_neurons=K,
        aux_k=aux_k,
        multi_k=multi_k,
        dead_neuron_threshold_steps=dead_neuron_threshold_steps,
    ).to(device)
    
    sae.fit(
        X_train=X,
        X_val=X_val,
        save_dir=checkpoint_dir,
        batch_size=batch_size,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        aux_coef=aux_coef,
        multi_coef=multi_coef,
        patience=patience,
        clip_grad=clip_grad,
    )

    return sae

def interpret_sae(
    texts: List[str],
    embeddings: Union[List, np.ndarray],
    sae: Union[SparseAutoencoder, List[SparseAutoencoder]],
    *,
    neuron_indices: Optional[List[int]] = None,
    n_random_neurons: Optional[int] = None,
    interpreter_model: str = "gpt-4o",
    annotator_model: str = "gpt-4o-mini",
    n_examples_for_interpretation: int = 20,
    max_words_per_example: int = 256,
    interpret_temperature: float = 0.7,
    max_interpretation_tokens: int = 50,
    n_candidates: int = 1,
    print_examples: int = 3,
    task_specific_instructions: Optional[str] = None,
) -> Dict:
    """Interpret neurons in a Sparse Autoencoder.
    
    Args:
        texts: Input text examples
        embeddings: Pre-computed embeddings for the input texts
        sae: A single SAE or a list of SAEs
        neuron_indices: Specific neuron indices to interpret (mutually exclusive with n_random_neurons)
        n_random_neurons: Number of random neurons to interpret (mutually exclusive with neuron_indices)
        interpreter_model: LLM to use for generating interpretations
        annotator_model: LLM to use for scoring interpretations
        n_examples: Number of examples to use for interpretation
        max_words_per_example: Maximum words per text to prompt the interpreter LLM with
        temperature: Temperature for LLM generation
        max_interpretation_tokens: Maximum tokens for interpretation
        n_candidates: Number of candidate interpretations per neuron
        print_examples: Number of top activating examples to print (0 to disable)
        task_specific_instructions: Optional task-specific instructions to include in the interpretation prompt
        
    Returns:
        Dictionary mapping neuron indices to their interpretations and top examples
    """
    if neuron_indices is None and n_random_neurons is None:
        raise ValueError("Either neuron_indices or n_random_neurons must be provided")
    
    if neuron_indices is not None and n_random_neurons is not None:
        raise ValueError("Only one of neuron_indices or n_random_neurons should be provided")
    
    embeddings = np.array(embeddings)
    X = torch.tensor(embeddings, dtype=torch.float32).to(device)
    
    # Convert single SAE to list for consistent handling
    if not isinstance(sae, list):
        sae = [sae]
    
    # Get activations from SAE(s)
    activations_list = []
    neuron_source_sae_info = []
    for s in sae:
        activations_list.append(s.get_activations(X))
        neuron_source_sae_info += [(s.m_total_neurons, s.k_active_neurons)] * s.m_total_neurons
    activations = np.concatenate(activations_list, axis=1)
    
    print(f"Activations shape (from {len(sae)} SAEs): {activations.shape}")
    
    # Select neurons to interpret
    if neuron_indices is None:
        total_neurons = activations.shape[1]
        neuron_indices = np.random.choice(total_neurons, size=n_random_neurons, replace=False)
    
    # Set up interpreter
    interpreter = NeuronInterpreter(
        interpreter_model=interpreter_model,
        annotator_model=annotator_model,
    )

    interpret_config = InterpretConfig(
        sampling=SamplingConfig(
            n_examples=n_examples_for_interpretation,
            max_words_per_example=max_words_per_example,
        ),
        llm=LLMConfig(
            temperature=interpret_temperature,
            max_interpretation_tokens=max_interpretation_tokens,
        ),
        n_candidates=n_candidates,
        task_specific_instructions=task_specific_instructions,
    )

    # Get interpretations
    interpretations = interpreter.interpret_neurons(
        texts=texts,
        activations=activations,
        neuron_indices=neuron_indices,
        config=interpret_config,
    )

    # Find top activating examples for each neuron if requested
    results_list = []
    for idx in neuron_indices:
        neuron_activations = activations[:, idx]
        result_dict = {
            "neuron_idx": int(idx),
            "source_sae": neuron_source_sae_info[idx],
            "interpretation": interpretations[idx][0] if n_candidates == 1 else interpretations[idx]
        }
        
        if print_examples > 0:
            top_indices = np.argsort(neuron_activations)[-print_examples:][::-1]
            top_examples = [texts[i] for i in top_indices]
            print(f"\nNeuron {idx} (from SAE M={neuron_source_sae_info[idx][0]}, K={neuron_source_sae_info[idx][1]}): {interpretations[idx][0]}")
            print(f"\nTop activating examples:")
            for i, example in enumerate(top_examples, 1):
                print(f"{i}. {get_text_for_printing(example, max_chars=256)}...")
                result_dict[f"top_example_{i}"] = example
            print("-"*100)
                
        results_list.append(result_dict)

    return pd.DataFrame(results_list)

def generate_hypotheses(
    texts: List[str],
    labels: Union[List[int], List[float], np.ndarray],
    embeddings: Union[List, np.ndarray],
    sae: Union[SparseAutoencoder, List[SparseAutoencoder]],
    cache_name: str,
    group_ids: Optional[np.ndarray] = None,
    *,
    classification: Optional[bool] = None, 
    selection_method: str = "separation_score",
    n_selected_neurons: int = 20,
    interpreter_model: str = "gpt-4o",
    annotator_model: str = "gpt-4o-mini",
    n_examples_for_interpretation: int = 20,
    max_words_per_example: int = 256,
    interpret_temperature: float = 0.7,
    max_interpretation_tokens: int = 50,
    n_candidate_interpretations: int = 1,
    filter: bool = False,
    n_scoring_examples: int = 100,
    scoring_metric: str = "f1",
    n_workers_interpretation: int = 10,
    n_workers_annotation: int = 30,
    interpretation_sampling_function: Callable = sample_top_zero,
    interpretation_sampling_kwargs: Dict[str, Any] = None,
    scoring_sampling_function: Callable = sample_top_zero,
    scoring_sampling_kwargs: Dict[str, Any] = None,
    task_specific_instructions: Optional[str] = None,
) -> pd.DataFrame:
    interpretation_sampling_kwargs = interpretation_sampling_kwargs or {}
    scoring_sampling_kwargs = scoring_sampling_kwargs or {}
    labels = np.array(labels)
    embeddings = np.array(embeddings)
    X = torch.tensor(embeddings, dtype=torch.float32).to(device)
    if classification is None:
        classification = np.all(np.isin(np.random.choice(labels, size=1000, replace=True), [0, 1]))
    if not isinstance(sae, list):
        sae = [sae]
    activations_list = []
    neuron_source_sae_info = []
    for s in sae:
        activations_list.append(s.get_activations(X))
        neuron_source_sae_info += [(s.m_total_neurons, s.k_active_neurons)] * s.m_total_neurons
    activations = np.concatenate(activations_list, axis=1)
    if n_selected_neurons > activations.shape[1]:
        raise ValueError(f"n_selected_neurons ({n_selected_neurons}) > total neurons ({activations.shape[1]})")
    extra_args = {}
    if group_ids is not None:
        extra_args["group_ids"] = group_ids
    selected_neurons, scores = select_neurons(
        activations=activations,
        target=labels,
        n_select=n_selected_neurons,
        method=selection_method,
        classification=classification,
        **extra_args
    )
    interpreter = NeuronInterpreter(
        cache_name=cache_name,
        interpreter_model=interpreter_model,
        annotator_model=annotator_model,
        n_workers_interpretation=n_workers_interpretation,
        n_workers_annotation=n_workers_annotation,
    )
    sampling_function = interpretation_sampling_function
    interpret_config = InterpretConfig(
        sampling=SamplingConfig(
            function=sampling_function,
            n_examples=n_examples_for_interpretation,
            max_words_per_example=max_words_per_example,
            extra_kwargs=interpretation_sampling_kwargs,
        ),
        llm=LLMConfig(
            temperature=interpret_temperature,
            max_interpretation_tokens=max_interpretation_tokens,
        ),
        n_candidates=n_candidate_interpretations,
        task_specific_instructions=task_specific_instructions,
    )
    interpretations = interpreter.interpret_neurons(
        texts=texts,
        activations=activations,
        neuron_indices=selected_neurons,
        config=interpret_config,
    )
    client = openai.OpenAI(api_key=os.environ["OPENAI_KEY_SAE"])
    def mentions_relevant_feature(text: str) -> bool:
        user_prompt = (
            "Please review the following text and determine whether it describes a feature related to:\n"
            "  1. Teacher or student behaviors (for example, working in teams or going to the bathroom),\n"
            "  2. Speech patterns (such as the use of a specific word or phrase), or\n"
            "  3. Aspects of a teacher's teaching style (e.g., calling on students or a specific method of teaching a mathematical concept).\n\n"
            "In contrast, the text should NOT be describing specific mathematical concepts (like fractions or long division) "
            "or physical classroom objects (such as boxes or windows). Importantly, while specific mathematical concepts (e.g., fractions) should not be included, "
            "more abstract features (e.g., use of a particular mathematical strategy) should.\n\n"
            "Answer with 'yes' if the text is about behaviors, speech, or teaching style, and 'no' otherwise. "
            "If you are unsure, please default to 'yes.'\n\n"
            f"Text: {text}"
        )
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": user_prompt}]
        )
        answer = response.choices[0].message.content.strip().lower()
        return answer.startswith("yes")
    neuron_relevance: Dict[int, bool] = {}
    if filter:
        from concurrent.futures import ThreadPoolExecutor
        tasks = [(idx, interp) for idx, interp_list in interpretations.items() for interp in interp_list]
        relevance_map: Dict[int, Dict[str, bool]] = {}
        def _check(task):
            idx, text = task
            try:
                ok = mentions_relevant_feature(text)
            except:
                ok = False
            return idx, text, ok
        with ThreadPoolExecutor(max_workers=100) as executor:
            for idx, text, ok in executor.map(_check, tasks):
                relevance_map.setdefault(idx, {})[text] = ok
        neuron_relevance = {idx: any(flags.values()) for idx, flags in relevance_map.items()}
    results = []
    if n_scoring_examples == 0:
        for idx, score in zip(selected_neurons, scores):
            row = {
                'neuron_idx': idx,
                'source_sae': neuron_source_sae_info[idx],
                f'target_{selection_method}': score,
                'mentions_relevant_feature': bool(filter and neuron_relevance.get(idx, False)),
                'best_interpretation': None,
                f'{scoring_metric}_fidelity_score': None
            }
            for j, text in enumerate(interpretations[idx], start=1):
                row[f'interpretation_{j}'] = text
                row[f'f1_score_{j}'] = None
            for j in range(len(interpretations[idx]) + 1, n_candidate_interpretations + 1):
                row[f'interpretation_{j}'] = None
                row[f'f1_score_{j}'] = None
            results.append(row)
    else:
        scoring_config = ScoringConfig(
            n_examples=n_scoring_examples,
            sampling_function=scoring_sampling_function,
            sampling_kwargs=scoring_sampling_kwargs,
        )
        if filter:
            scored_interpretations = {
                idx: interpretations[idx]
                for idx in interpretations
                if neuron_relevance.get(idx, False)
            }
        else:
            scored_interpretations = interpretations
        metrics = interpreter.score_interpretations(
            texts=texts,
            activations=activations,
            interpretations=scored_interpretations,
            config=scoring_config
        )
        for idx, score in zip(selected_neurons, scores):
            is_rel = not filter or neuron_relevance.get(idx, False)
            row = {
                'neuron_idx': idx,
                'source_sae': neuron_source_sae_info[idx],
                f'target_{selection_method}': score,
                'mentions_relevant_feature': bool(filter and neuron_relevance.get(idx, False))
            }
            for j, interp in enumerate(interpretations[idx], start=1):
                row[f'interpretation_{j}'] = interp
                if is_rel:
                    row[f'f1_score_{j}'] = metrics[idx][interp][scoring_metric]
                else:
                    row[f'f1_score_{j}'] = None
            for j in range(len(interpretations[idx]) + 1, n_candidate_interpretations + 1):
                row[f'interpretation_{j}'] = None
                row[f'f1_score_{j}'] = None
            if is_rel:
                best = max(
                    interpretations[idx],
                    key=lambda interp: metrics[idx][interp][scoring_metric]
                )
                row['best_interpretation'] = best
                row[f'{scoring_metric}_fidelity_score'] = metrics[idx][best][scoring_metric]
            else:
                row['best_interpretation'] = None
                row[f'{scoring_metric}_fidelity_score'] = None
            results.append(row)
    return pd.DataFrame(results)

def evaluate_hypotheses(
    hypotheses_df: pd.DataFrame,
    texts: List[str],
    labels: Union[List[int], List[float], np.ndarray],
    cache_name: str,
    *,
    annotator_model: str = "gpt-4o-mini",
    max_words_per_example: int = 256,
    classification: Optional[bool] = None,
    n_workers_annotation: int = 30,
    corrected_pval_threshold: float = 0.1,
) -> pd.DataFrame:
    """Evaluate hypotheses on a heldout dataset.
    
    Args:
        hypotheses_df: DataFrame from generate_hypotheses()
        texts: Heldout text examples
        labels: Heldout labels
        annotator_model: Model to use for annotation
        max_words_per_example: Maximum words per example for annotation
        classification: Whether this is a classification task. If None, inferred from labels
        dataset_name: Name for caching
        
    Returns:
        DataFrame with original columns plus evaluation metrics
    """
    labels = np.array(labels)
    
    # Infer classification if not specified
    if classification is None:
        classification = np.all(np.isin(np.random.choice(labels, size=1000, replace=True), [0, 1]))

    # Extract hypotheses from dataframe
    hypotheses = hypotheses_df['interpretation'].tolist()
    
    # Step 1: Get annotations for each hypothesis on the texts
    print(f"Step 1: Annotating texts with {len(hypotheses)} hypotheses")
    hypothesis_annotations = annotate_texts_with_concepts(
        texts=texts,
        concepts=hypotheses,
        max_words_per_example=max_words_per_example,
        model=annotator_model,
        cache_name=cache_name,
        n_workers=n_workers_annotation,
    )
    
    # Step 2: Evaluate annotations against the true labels
    print("Step 2: Computing predictiveness of hypothesis annotations")
    metrics, evaluation_df = score_hypotheses(
        hypothesis_annotations=hypothesis_annotations,
        y_true=np.array(labels),
        classification=classification,
        corrected_pval_threshold=corrected_pval_threshold,
    )
    
    return metrics, evaluation_df
