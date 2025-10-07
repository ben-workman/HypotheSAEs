"""Methods for interpreting SAE neurons using LLMs."""

import numpy as np
from typing import List, Dict, Optional, Tuple, Callable, Any
from tqdm.auto import tqdm
import concurrent.futures
import os 
from dataclasses import dataclass, field

from .llm_api import get_completion
from .utils import load_prompt, truncate_text
from .annotate import annotate, CACHE_DIR

DEFAULT_TASK_SPECIFIC_INSTRUCTIONS = """An example feature could be:
- "uses multiple adjectives to describe colors"
- "describes a patient experiencing seizures or epilepsy"
- "contains multiple single-digit numbers\""""

def sample_top_zero(
    texts,
    activations,
    neuron_idx,
    n_examples,
    max_words_per_example=None,
    random_seed=None
):
    if random_seed is not None:
        np.random.seed(random_seed)
    acts = activations[:, neuron_idx]
    n_per = max(1, n_examples // 2)

    pos_idx_all = np.where(acts > 0)[0]
    if len(pos_idx_all) == 0:
        top_idx = np.argsort(acts)[-n_per:]
    else:
        sel = min(n_per, len(pos_idx_all))
        order = np.argsort(acts[pos_idx_all])
        top_idx = pos_idx_all[order][-sel:]

    zero_idx_all = np.where(acts == 0)[0]
    if len(zero_idx_all) >= n_per:
        neg_idx = np.random.choice(zero_idx_all, size=n_per, replace=False)
    elif len(zero_idx_all) > 0:
        neg_idx = zero_idx_all
    else:
        asc = np.argsort(acts)
        mask = np.ones(len(acts), dtype=bool)
        mask[top_idx] = False
        asc = asc[mask[asc]]
        neg_idx = asc[:n_per]

    pos_texts = [texts[i] for i in top_idx]
    neg_texts = [texts[i] for i in neg_idx]
    if max_words_per_example:
        pos_texts = [truncate_text(t, max_words_per_example) for t in pos_texts]
        neg_texts = [truncate_text(t, max_words_per_example) for t in neg_texts]

    return {
        "positive_texts": pos_texts,
        "negative_texts": neg_texts,
        "positive_activations": acts[top_idx].tolist(),
        "negative_activations": acts[neg_idx].tolist()
    }

def sample_percentile_bins(
    texts,
    activations,
    neuron_idx,
    n_examples,
    max_words_per_example=None,
    high_percentile=(90, 100),
    low_percentile=None,
    random_seed=None
):
    if random_seed is not None:
        np.random.seed(random_seed)
    acts = activations[:, neuron_idx]
    n_per = max(1, n_examples // 2)

    pos_mask = acts > 0
    pos_vals = acts[pos_mask]
    pos_indices = np.where(pos_mask)[0]

    if len(pos_vals) > 0:
        lo_hi = np.percentile(pos_vals, high_percentile[0])
        hi_hi = np.percentile(pos_vals, high_percentile[1])
        high_mask = (pos_vals >= lo_hi) & (pos_vals <= hi_hi)
        high_indices = pos_indices[high_mask]
        if len(high_indices) >= n_per:
            high_sample_indices = np.random.choice(high_indices, size=n_per, replace=False)
        elif len(high_indices) > 0:
            high_sample_indices = high_indices
        else:
            order = np.argsort(pos_vals)
            sel = min(n_per, len(pos_vals))
            high_sample_indices = pos_indices[order][-sel:]
    else:
        sel = min(n_per, len(acts))
        high_sample_indices = np.argsort(acts)[-sel:]

    if low_percentile is not None and len(pos_vals) > 0:
        lo_lo = np.percentile(pos_vals, low_percentile[0])
        hi_lo = np.percentile(pos_vals, low_percentile[1])
        low_mask = (pos_vals >= lo_lo) & (pos_vals <= hi_lo)
        low_indices = pos_indices[low_mask]
    else:
        low_indices = np.where(acts == 0)[0]

    if len(low_indices) >= n_per:
        low_sample_indices = np.random.choice(low_indices, size=n_per, replace=False)
    elif len(low_indices) > 0:
        low_sample_indices = low_indices
    else:
        asc = np.argsort(acts)
        used = set(high_sample_indices.tolist())
        low_sample_indices = [i for i in asc if i not in used][:n_per]
        low_sample_indices = np.array(low_sample_indices, dtype=int)

    pos_texts = [texts[i] for i in high_sample_indices]
    neg_texts = [texts[i] for i in low_sample_indices]
    if max_words_per_example:
        pos_texts = [truncate_text(t, max_words_per_example) for t in pos_texts]
        neg_texts = [truncate_text(t, max_words_per_example) for t in neg_texts]

    return {
        "positive_texts": pos_texts,
        "negative_texts": neg_texts,
        "positive_activations": acts[high_sample_indices].tolist(),
        "negative_activations": acts[low_sample_indices].tolist()
    }

def sample_custom(
    texts: List[str],
    activations: np.ndarray,
    neuron_idx: int,
    random_seed: Optional[int] = None
) -> Dict[str, List[str]]:
    """Sample examples using a custom function.
    
    This function should return a dictionary with keys that correspond to your prompt template.
    The default prompt template is "interpret-neuron-binary.txt", which expects two keys: positive_samples and negative_samples.

    For example, you can write a custom sampling function that outputs:
    - only top-activating examples
    - top-activating, medium-activating, and non-activating examples
    - etc.

    Note that if you change the sampling setup, you will also need to write a new prompt template.
    Ensure that the keys in your output dictionary match the keys in your prompt template.
    
    Args:
        texts: List of all text examples
        activations: Neuron activation matrix (n_samples, n_neurons)
        neuron_idx: Index of neuron to sample examples for
        [any other arguments]
    """
    pass

@dataclass
class SamplingConfig:
    function: Callable = sample_top_zero
    n_examples: int = 20 # Number of examples to sample to prompt the interpreter
    random_seed: Optional[int] = None # Random seed for example sampling
    max_words_per_example: Optional[int] = 256 # Maximum number of words per text example, truncated if necessary
    extra_kwargs: Dict[str, Any] = field(default_factory=dict) # Extra keyword arguments for the sampling function

@dataclass
class LLMConfig:
    temperature: float = 0.7 # Temperature for the interpreter model
    max_interpretation_tokens: int = 100 # Maximum number of tokens for each generated interpretation
    timeout: float = 10.0 # Timeout for the interpreter model (in seconds)

@dataclass
class InterpretConfig:
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    n_candidates: int = 1 # Number of candidate interpretations per neuron
    prompt_name: str = "interpret-neuron-binary" # Name of the prompt template file to use
    task_specific_instructions: str = DEFAULT_TASK_SPECIFIC_INSTRUCTIONS # Task-specific instructions for the interpreter model

@dataclass
class ScoringConfig:
    n_examples: int = 100 # Number of examples to score interpretation fidelity (half top-activating, half zero-activating)
    max_words_per_example: Optional[int] = 256 # Maximum number of words per text example, truncated if necessary
    sampling_function: Callable = sample_top_zero # Function to sample examples for scoring
    sampling_kwargs: Dict[str, Any] = field(default_factory=dict) # Extra keyword arguments for the sampling function

class NeuronInterpreter:
    def __init__(
        self,
        interpreter_model: str = "gpt-4o",
        annotator_model: str = "gpt-4o-mini",
        n_workers_interpretation: int = 10,
        n_workers_annotation: int = 30,
        cache_name: Optional[str] = None,
    ):
        """Initialize a NeuronInterpreter."""
        self.interpreter_model = interpreter_model
        self.annotator_model = annotator_model
        self.n_workers_interpretation = n_workers_interpretation
        self.n_workers_annotation = n_workers_annotation
        self.cache_name = cache_name
        
    def _get_interpretation_completion(
        self,
        prompt_template: str,
        formatted_examples: dict,
        config: InterpretConfig,
    ) -> str:
        """Get and parse interpretation completion from LLM."""
        prompt = prompt_template.format(
            task_specific_instructions=config.task_specific_instructions,
            **formatted_examples
        )
        
        kwargs = {
            "model": self.interpreter_model,
            "prompt": prompt,
            "max_tokens": config.llm.max_interpretation_tokens,
            "timeout": config.llm.timeout
        }

        if self.interpreter_model.startswith('o'):
            kwargs["reasoning_effort"] = "low"
        else:
            kwargs["temperature"] = config.llm.temperature

        response = get_completion(**kwargs)
        
        return self._parse_interpretation(response)
    
    def _parse_interpretation(self, response: str) -> str:
        """Parse raw LLM response into clean interpretation string."""
        response = response.strip()
        if response.startswith('- '):
            response = response[2:]
        if response.startswith('"-'):
            response = response[2:]
        if response.startswith('" -'):
            response = response[3:]
        return response.strip('"')

    def interpret_neuron(
        self,
        texts: List[str],
        activations: np.ndarray,
        neuron_idx: int,
        config: InterpretConfig
    ) -> str:
        """Generate interpretation for a single neuron."""
        sampling_kwargs = dict(config.sampling.extra_kwargs)
        if config.sampling.random_seed is not None:
            sampling_kwargs.setdefault("random_seed", config.sampling.random_seed)

        formatted_examples = config.sampling.function(
            texts=texts,
            activations=activations,
            neuron_idx=neuron_idx,
            n_examples=config.sampling.n_examples,
            max_words_per_example=config.sampling.max_words_per_example,
            **sampling_kwargs
        )

        prompt_template = load_prompt(config.prompt_name)
        return self._get_interpretation_completion(
            prompt_template=prompt_template,
            formatted_examples=formatted_examples,
            config=config
        )

    def interpret_neurons(
        self,
        texts,
        activations,
        neuron_indices,
        config=None
    ):
        config = config or InterpretConfig()
        interpretations = {idx: [None] * config.n_candidates for idx in neuron_indices}
        tasks = []
        for neuron_idx in neuron_indices:
            for candidate_idx in range(config.n_candidates):
                rs = None if config.sampling.random_seed is None else config.sampling.random_seed + candidate_idx
                cand_sampling = SamplingConfig(
                    function=config.sampling.function,
                    n_examples=config.sampling.n_examples,
                    random_seed=rs,
                    max_words_per_example=config.sampling.max_words_per_example,
                    extra_kwargs=dict(config.sampling.extra_kwargs)
                )
                cand_config = InterpretConfig(
                    sampling=cand_sampling,
                    llm=config.llm,
                    n_candidates=1,
                    prompt_name=config.prompt_name,
                    task_specific_instructions=config.task_specific_instructions
                )
                tasks.append((neuron_idx, candidate_idx, cand_config))
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_workers_interpretation) as executor:
            future_to_key = {
                executor.submit(
                    self.interpret_neuron,
                    texts=texts,
                    activations=activations,
                    neuron_idx=neuron_idx,
                    config=cand_config
                ): (neuron_idx, candidate_idx)
                for neuron_idx, candidate_idx, cand_config in tasks
            }
            iterator = concurrent.futures.as_completed(future_to_key)
            iterator = tqdm(iterator, total=len(future_to_key),
                            desc=f"Generating {config.n_candidates} interpretation(s) per neuron")
            for fut in iterator:
                neuron_idx, candidate_idx = future_to_key[fut]
                try:
                    interpretations[neuron_idx][candidate_idx] = fut.result()
                except Exception as e:
                    print(f"Failed to generate interpretation {candidate_idx} for neuron {neuron_idx}: {e}")
        return interpretations

    def _compute_metrics(
        self,
        annotations: np.ndarray,
        labels: np.ndarray,
        activations: np.ndarray
    ) -> Dict[str, float]:
        """Compute evaluation metrics for a single interpretation."""
        true_pos = np.mean(annotations[labels == 1])
        false_pos = np.mean(annotations[labels == 0])
        
        return {
            "recall": true_pos,
            "precision": 1 - false_pos,
            "f1": 2 * true_pos * (1 - false_pos) / (true_pos + (1 - false_pos)),
            "correlation": np.corrcoef(activations, annotations)[0,1]
        }

    def score_interpretations(
        self,
        texts: List[str],
        activations: np.ndarray,
        interpretations: Dict[int, List[str]],
        config: Optional[ScoringConfig] = None
    ) -> Dict[int, Dict[str, Dict[str, float]]]:
        """Score all interpretations for all neurons."""
        config = config or ScoringConfig()
        tasks = []
        scoring_info = {}

        for neuron_idx, neuron_interps in interpretations.items():
            formatted_examples = config.sampling_function(
                texts=texts,
                activations=activations,
                neuron_idx=neuron_idx,
                n_examples=config.n_examples,
                max_words_per_example=config.max_words_per_example,
                random_seed=neuron_idx,  # Deterministic seed based on neuron_idx
                **config.sampling_kwargs
            )
            
            eval_texts = formatted_examples["positive_texts"] + formatted_examples["negative_texts"]
            scoring_info[neuron_idx] = {
                'texts': eval_texts,
                'activations': formatted_examples["positive_activations"] + formatted_examples["negative_activations"],
                'binarized_activations': np.concatenate([
                    np.ones(len(formatted_examples["positive_texts"])),
                    np.zeros(len(formatted_examples["negative_texts"]))
                ])
            }

            for interp in neuron_interps:
                for text in eval_texts:
                    tasks.append((text, interp))

        # Annotate all tasks
        progress_desc = f"Scoring neuron interpretation fidelity ({len(interpretations)} neurons; {len(next(iter(interpretations.values())))} candidate interps per neuron; {config.n_examples} examples to score each interp)"
        
        cache_path = None if self.cache_name is None else os.path.join(CACHE_DIR, f"{self.cache_name}_interp-scoring.json")
        annotations = annotate(
            tasks=tasks,
            cache_path=cache_path,
            n_workers=self.n_workers_annotation,
            show_progress=True,
            model=self.annotator_model,
            progress_desc=progress_desc
        )

        # Compute metrics for all interpretations
        all_metrics = {}
        for neuron_idx, neuron_interps in interpretations.items():
            all_metrics[neuron_idx] = {}
            neuron_scoring_info = scoring_info[neuron_idx]

            for interp in neuron_interps:
                annot = [annotations[interp][text] for text in neuron_scoring_info['texts']]
                all_metrics[neuron_idx][interp] = self._compute_metrics(
                    annotations=np.array(annot),
                    labels=neuron_scoring_info['binarized_activations'],
                    activations=neuron_scoring_info['activations']
                )

        return all_metrics
        