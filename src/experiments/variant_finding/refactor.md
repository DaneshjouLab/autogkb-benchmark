# Variant Extraction Refactor

## Goal
The goal of the module is to be able to take in an article/pmcid and return a list of variants found in the article. This should then be compared
against the ground truth variants (from the benchmark) to evaluate the performance of the variant extraction method. Recall is the primary metric.

## Current State

The module has 5 extraction approaches spread across 4 subdirectories, each with its own standalone script, duplicated helper functions, inconsistent output formats, and independent evaluation logic:

- **just_ask/** - LLM-only extraction (has its own `score_variants`, `load_benchmark_variants`, `extract_json_array`)
- **regex_variants/** - Regex extraction v1-v5 (has `extract_all_variants`, `normalize_*`, `get_combined_text`)
- **regex_term_norm/** - Copy-pasted all of regex v5's code + adds normalization layer
- **regex_llm_filter/** - Imports from regex v5 but has its own `extract_json_array`, `load_prompts`
- **pubtator/** - API-based extraction (uses `benchmark_v2.variant_bench` for scoring, unlike just_ask)

### Key Problems
1. **Massive code duplication**: `regex_term_norm` copy-pastes ~300 lines from `regex_variants/extract_variants_v5.py` (all extraction functions, normalization helpers, gene lists)
2. **Inconsistent scoring**: `just_ask` has its own `score_variants` and `load_benchmark_variants` instead of using `benchmark_v2.variant_bench`
3. **Inconsistent output formats**: Each script structures its results JSON differently (different key names, different nesting)
4. **No shared interface**: `VariantExtractor` base class exists but nothing implements it
5. **Duplicated `extract_json_array`**: Both `just_ask` and `regex_llm_filter` define their own version
6. **Duplicated `load_prompts`**: Both `just_ask` and `regex_llm_filter` define their own version
7. **Old experimental versions**: `regex_variants/` contains v1-v4 scripts that are no longer used

---

## Changes

### 1. Define the VariantExtractor Interface (`VariantExtractor.py`)

Update the existing base class to define the full contract:

```python
class VariantExtractor:
    name: str  # e.g. "just_ask", "regex_v5", "regex_llm_filter", "pubtator", "regex_term_norm"

    def __init__(self, **kwargs):
        ...

    def get_variants(self, pmcid: str) -> list[str]:
        """Extract variants from an article. Returns list of variant strings."""
        raise NotImplementedError
```

- `get_variants(pmcid) -> list[str]` is the only required method
- Constructor takes keyword args for configuration (model, prompt_version, etc.)
- The `name` attribute identifies the extractor for results tracking
- Remove `get_pmcid_markdown_text` from the base class — text retrieval is an implementation detail

### 2. Create Shared Utilities (`utils.py`)

Create `src/experiments/variant_finding/utils.py` with functions currently duplicated across scripts:

**Move from `regex_variants/extract_variants_v5.py` (single source of truth for all regex-based extractors):**
- `extract_all_variants(text) -> list[str]`
- `extract_rsids(text) -> list[str]`
- `extract_snp_notations(text) -> list[str]`
- `extract_star_alleles(text) -> list[str]`
- `extract_hla_alleles(text) -> list[str]`
- `normalize_hla(variant) -> str`
- `normalize_star_allele(gene, allele_num) -> str`
- `get_combined_text(pmcid) -> tuple[str, str | None]`
- `get_snp_expander() -> SNPExpander` (singleton)
- The `pgx_genes` list (currently duplicated in extract_star_alleles — define once as a module constant)

**Move from `just_ask/just_ask.py` and `regex_llm_filter/regex_llm_filter.py`:**
- `extract_json_array(text) -> list[str]` — merge the two implementations (the `regex_llm_filter` version is better; it handles markdown code blocks)
- `load_prompts(prompts_file: Path) -> dict` — generic prompt loader from YAML

### 3. Implement Extractor Classes

Each extractor goes in its own file within its subdirectory and implements `VariantExtractor`:

#### `just_ask/just_ask.py` → `JustAskExtractor`
```python
class JustAskExtractor(VariantExtractor):
    name = "just_ask"
    def __init__(self, model: str, prompt_version: str = "v1"):
        self.model = model
        self.prompt_version = prompt_version
        self.prompts = load_prompts(Path(__file__).parent / "prompts.yaml")
    def get_variants(self, pmcid: str) -> list[str]:
        # Use get_methods_and_conclusions_text, call_llm, extract_json_array
        ...
```
- Remove the local `score_variants` and `load_benchmark_variants` — use `benchmark_v2.variant_bench` instead
- Remove the local `extract_json_array` — use shared utils

#### `regex_variants/regex_extractor.py` → `RegexExtractor`
```python
class RegexExtractor(VariantExtractor):
    name = "regex_v5"
    def __init__(self):
        get_snp_expander()  # pre-initialize
    def get_variants(self, pmcid: str) -> list[str]:
        combined_text, _ = get_combined_text(pmcid)
        return extract_all_variants(combined_text)
```
- All extraction logic lives in shared `utils.py`; this class is thin

#### `regex_llm_filter/regex_llm_filter.py` → `RegexLLMFilterExtractor`
```python
class RegexLLMFilterExtractor(VariantExtractor):
    name = "regex_llm_filter"
    def __init__(self, model: str, prompt_version: str = "v1"):
        ...
    def get_variants(self, pmcid: str) -> list[str]:
        # Step 1: regex extraction via shared utils
        # Step 2: LLM filtering
        ...
```
- Remove `sys.path.append` hack — imports should work via package imports
- Remove local `extract_json_array` and `load_prompts` — use shared utils

#### `regex_term_norm/regex_term_norm.py` → `RegexTermNormExtractor`
```python
class RegexTermNormExtractor(VariantExtractor):
    name = "regex_term_norm"
    def __init__(self, threshold=0.8, min_score=0.9, top_k=3):
        ...
    def get_variants(self, pmcid: str) -> list[str]:
        # Step 1: regex extraction via shared utils (NOT copy-pasted code)
        # Step 2: normalization via VariantLookup
        ...
```
- **Delete all copy-pasted extraction functions** — import from shared `utils.py`
- Keep only the normalization logic (`normalize_variants`, `apply_normalization_to_variants`)

#### `pubtator/pubtator_extractor.py` → `PubTatorExtractor`
```python
class PubTatorExtractor(VariantExtractor):
    name = "pubtator"
    def __init__(self, full_text: bool = True):
        ...
    def get_variants(self, pmcid: str) -> list[str]:
        # fetch from PubTator API, extract variants from BioC JSON
        ...
```
- Keep PubTator-specific logic (API calls, BioC parsing) local to this class
- Rate limiting is an internal implementation detail

### 4. Create Factory Function

Add to `VariantExtractor.py` or a new `factory.py`:

```python
def create_variant_extractor(method: str, **kwargs) -> VariantExtractor:
    """Factory function to create a variant extractor by name.

    Args:
        method: One of "just_ask", "regex_v5", "regex_llm_filter", "regex_term_norm", "pubtator"
        **kwargs: Passed to the extractor constructor (e.g. model, prompt_version)
    """
    extractors = {
        "just_ask": JustAskExtractor,
        "regex_v5": RegexExtractor,
        "regex_llm_filter": RegexLLMFilterExtractor,
        "regex_term_norm": RegexTermNormExtractor,
        "pubtator": PubTatorExtractor,
    }
    ...
```

### 5. Consolidated Evaluation (`eval.py`)

Create `src/experiments/variant_finding/eval.py`:

```python
def evaluate_extractor(
    extractor: VariantExtractor,
    pmcids: list[str] | None = None,
    max_articles: int | None = None,
) -> dict:
    """Run an extractor against the benchmark and return standardized results."""
```

This function:
- Loads benchmark data via `load_variant_bench_data()`
- Calls `extractor.get_variants(pmcid)` for each article
- Scores via `benchmark_v2.variant_bench.score_variants()`
- Prints progress (status symbols, per-article recall/precision)
- Returns standardized results dict (see format below)
- Saves results to `results/{extractor.name}_{timestamp}.json`

**Standardized results format:**
```json
{
    "extractor": "regex_v5",
    "config": {},
    "timestamp": "ISO 8601",
    "articles_processed": 32,
    "avg_recall": 0.934,
    "avg_precision": 0.419,
    "perfect_recall_count": 25,
    "per_article_results": [
        {
            "pmcid": "PMC5508045",
            "recall": 1.0,
            "precision": 0.8,
            "true_count": 5,
            "extracted_count": 6,
            "matches": [...],
            "misses": [...],
            "extras": [...]
        }
    ]
}
```

All extractors use this same output format — no more per-method result structures.

### 6. Unified CLI Runner (`run.py`)

Create `src/experiments/variant_finding/run.py` as the single entry point:

```bash
# Run any extractor from one CLI
python -m src.experiments.variant_finding.run --method regex_v5
python -m src.experiments.variant_finding.run --method just_ask --model claude-opus-4-5-20251101 --prompt v3
python -m src.experiments.variant_finding.run --method regex_llm_filter --model gpt-4o --prompt v1 --max-articles 5
python -m src.experiments.variant_finding.run --method pubtator
```

This replaces the individual `if __name__ == "__main__"` blocks in each script.

### 7. Clean Up Old Files

- Move `regex_variants/extract_variants_v{1,2,3,4}.py` to `regex_variants/old_results/`
- Move all existing `results*.json`, `misses_analysis*.md` files to their respective `old_results/` directories
- Remove the old standalone `run_experiment()` functions from each file (replaced by `eval.py`)
- Delete `regex_term_norm/extract_variants_term_norm.py` after extracting normalization logic (it's 100% copy-paste + normalization)
- Remove `just_ask`'s local `score_variants` and `load_benchmark_variants`

### 8. File Structure After Refactor

```
src/experiments/variant_finding/
├── VariantExtractor.py          # Base class + factory function
├── utils.py                     # Shared extraction functions, helpers
├── eval.py                      # Consolidated evaluation logic
├── run.py                       # Unified CLI entry point
├── just_ask/
│   ├── just_ask.py              # JustAskExtractor class
│   ├── prompts.yaml             # Prompts (unchanged)
│   └── old_results/             # Previous result files
├── regex_variants/
│   ├── regex_extractor.py       # RegexExtractor class (thin wrapper)
│   └── old_results/             # v1-v4 scripts + old result files
├── regex_llm_filter/
│   ├── regex_llm_filter.py      # RegexLLMFilterExtractor class
│   ├── prompts.yaml             # Prompts (unchanged)
│   └── old_results/             # Previous result files
├── regex_term_norm/
│   ├── regex_term_norm.py       # RegexTermNormExtractor class
│   └── old_results/             # Previous result files
├── pubtator/
│   ├── pubtator_extractor.py    # PubTatorExtractor class
│   └── old_results/             # Previous result files
└── results/                     # All new results go here (shared)
```

---

## Implementation Order

1. Create `utils.py` — move shared extraction functions from `extract_variants_v5.py`
2. Update `VariantExtractor.py` — finalize interface + factory
3. Create `eval.py` — consolidated evaluation
4. Implement each extractor class (start with `RegexExtractor` since it's simplest, then `JustAskExtractor`, `PubTatorExtractor`, `RegexTermNormExtractor`, `RegexLLMFilterExtractor`)
5. Create `run.py` — unified CLI
6. Move old files to `old_results/` directories
7. Verify all extractors produce the same output format via `eval.py`