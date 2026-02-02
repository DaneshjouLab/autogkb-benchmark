# Pipeline & Eval Refactor Plan

## Problem

The experiments were refactored into a clean factory pattern (factory class + pure function methods + Pydantic models + unified CLI), but `generation_pipeline/pipeline.py` (909 lines) and `eval_pipeline/eval_pipeline.py` (824 lines) remain monolithic. They duplicate logic from the experiments, use stale import paths, and can't leverage the new method-swapping capabilities.

---

## Current State

### Experiments (refactored, good)
Each experiment module follows a consistent pattern:
- **Factory class** (`VariantExtractor`, `SentenceGenerator`, `CitationFinder`, `SummaryGenerator`) with lazy-loaded method registry
- **Pure function methods** in `methods/` directory
- **Pydantic output models** in `models.py`
- **Shared utils** in `utils.py` (parsing, text helpers)
- **Standalone eval** in `eval.py`
- **CLI runner** in `run.py`

### Pipeline (monolithic, needs refactor)
- `pipeline.py` hardcodes `regex_v5` imports directly — bypasses `VariantExtractor` factory entirely
- Inline `parse_sentence_output()` and `parse_citation_output()` duplicate logic from experiment `utils.py`
- Custom dataclasses (`PMCIDResult`, `VariantResult`) instead of experiment Pydantic models
- Config YAML claims method is configurable but the code ignores it
- `prompts.yaml` is a single monolithic file rather than per-method prompt files

### Eval Pipeline (monolithic, needs refactor)
- Reimplements F1 calculation (`calculate_f1_metrics`) instead of reusing `benchmark_v2`
- Custom dataclasses (`VariantEvaluationResult`, `SentenceEvaluationResult`) that don't align with experiment eval outputs
- `evaluate_sentences_for_pmcid()` rebuilds sentence scoring logic that `sentence_generation/eval.py` already handles
- Tightly coupled to the generation pipeline's output format

---

## Refactor Plan

### Phase 1: Make the pipeline delegate to experiment factory classes

**Goal:** Replace all inline stage logic in `pipeline.py` with calls to the existing experiment classes.

**Changes to `pipeline.py`:**

1. **Replace hardcoded variant extraction** (lines 75-79, 190-256) with:
   ```python
   from src.experiments.variant_finding.variant_extractor import VariantExtractor

   # In process_pmcid():
   extractor = VariantExtractor(method=config["variant_extraction"]["method"])
   result.variants = extractor.get_variants(pmcid)
   ```

2. **Replace inline sentence generation** (lines 264-381) with:
   ```python
   from src.experiments.sentence_generation.sentence_generator import SentenceGenerator

   generator = SentenceGenerator(method=config["sentence_generation"]["method"])
   output = generator.generate(pmcid, variants)
   ```

3. **Replace inline citation finding** (lines 389-534) with:
   ```python
   from src.experiments.citations.citation_finder import CitationFinder

   finder = CitationFinder(method=config["citation_finding"]["method"])
   citations = finder.find(pmcid, sentences)
   ```

4. **Replace inline summary generation** (lines 542-612) with:
   ```python
   from src.experiments.summary.summary_generator import SummaryGenerator

   summarizer = SummaryGenerator(method=config["summary_generation"]["method"])
   summary = summarizer.generate(pmcid, sentences)
   ```

5. **Delete all inline parsing functions** (`parse_sentence_output`, `parse_citation_output`, `format_associations_for_summary`, `format_citations_for_summary`) — these are already handled inside the experiment methods.

6. **Delete the stale import block** (lines 75-79) referencing the old `regex_variants/extract_variants_v5` path.

7. **Delete `initialize_variant_extraction()`** — the factory class handles initialization internally.

**Expected result:** `pipeline.py` drops from ~909 lines to ~300 lines. The orchestration logic (`process_pmcid`, `run_pipeline`, `main`) stays, but each stage is a 3-5 line delegation.

---

### Phase 2: Replace custom dataclasses with experiment Pydantic models

**Goal:** Use the same data models the experiments already define so pipeline outputs can be directly consumed by experiment eval.

1. **Delete `VariantResult` and `PMCIDResult` dataclasses** from `pipeline.py`.

2. **Define a thin `PipelineResult` Pydantic model** that composes the experiment output models:
   ```python
   from src.experiments.variant_finding.models import VariantExtractionOutput
   from src.experiments.sentence_generation.models import SentenceGenerationOutput
   from src.experiments.citations.models import CitationOutput
   from src.experiments.summary.models import SummaryOutput

   class PipelineResult(BaseModel):
       pmcid: str
       variants: VariantExtractionOutput | None = None
       sentences: SentenceGenerationOutput | None = None
       citations: CitationOutput | None = None
       summary: SummaryOutput | None = None
   ```

3. **Update `save_pmcid_result()`** to serialize the Pydantic model (`.model_dump()`).

4. **Update output JSON format** — the new format nests each stage's output under its key, making it clear which experiment method produced each result.

---

### Phase 3: Make config actually control method selection

**Goal:** The YAML config should be the single source of truth for which method each stage uses.

1. **Update config schema** to include a `method` key per stage that maps to the factory registry:
   ```yaml
   variant_extraction:
     method: regex_v5        # maps to VariantExtractor("regex_v5")

   sentence_generation:
     method: batch_judge_ask  # maps to SentenceGenerator("batch_judge_ask")
     model: gpt-4o

   citation_finding:
     method: one_shot_citations
     model: gpt-4o

   summary_generation:
     method: basic_summary
     model: gpt-4o
   ```

2. **Remove `prompts.yaml`** — prompts are already co-located with each method in `methods/prompts/`. The factory class + method function handles prompt loading internally.

3. **Pass method-specific kwargs from config** to the factory:
   ```python
   extractor = VariantExtractor(
       method=cfg["method"],
       **cfg.get("kwargs", {})
   )
   ```

---

### Phase 4: Consolidate eval pipeline to reuse experiment eval modules

**Goal:** `eval_pipeline.py` becomes a thin orchestrator that delegates to each experiment's `eval.py`.

1. **Delete `calculate_f1_metrics()`** — already exists in `benchmark_v2`.

2. **Delete `evaluate_variants_for_pmcid()`** — replace with:
   ```python
   from src.experiments.variant_finding.eval import evaluate_from_file
   # or evaluate inline results
   ```

3. **Delete `evaluate_sentences_for_pmcid()`** — replace with:
   ```python
   from src.experiments.sentence_generation.eval import evaluate_from_file
   ```

4. **Delete all custom eval dataclasses** (`VariantEvaluationResult`, `SentenceEvaluationResult`, `PMCIDEvaluationResult`, `VariantMetrics`, `EvaluationResult`) and use the result types from experiment `eval.py` modules.

5. **Keep `evaluate_pipeline_output()` and `main()`** as thin orchestration — iterate over PMCIDs, call the appropriate experiment eval function, aggregate results.

6. **Keep `generate_evaluation_summary()`** — this is eval-pipeline-specific and not duplicated.

**Expected result:** `eval_pipeline.py` drops from ~824 lines to ~250 lines.

---

### Phase 5: Enable eval-only mode on the generation pipeline

**Goal:** Allow running eval directly from pipeline outputs without a separate script.

1. **Add `--eval` flag to `pipeline.py`** that takes a previous output directory and runs evaluation on it:
   ```bash
   # Generate
   python -m src.generation_pipeline.pipeline --config configs/base.yaml

   # Evaluate a previous run
   python -m src.generation_pipeline.pipeline --eval outputs/base_config_20240101/
   ```

2. **Alternatively**, keep eval as a separate entry point but have it import from the experiment eval modules rather than reimplementing.

---

### Phase 6: Standardize run output directory structure

**Goal:** Each pipeline run produces a self-contained directory with all artifacts organized by stage, making runs reproducible and easy to navigate.

**Output structure:**
```
outputs/<run_name>/
   config.yaml          # Exact config used for this run (snapshot for reproducibility)
   metadata.yaml        # Run metadata (timestamp, PMCIDs processed, stages run, git sha, etc.)
   variants.json        # Aggregated variant extraction results across all PMCIDs
   sentences/           # Per-PMCID sentence generation outputs
   citations/           # Per-PMCID citation finding outputs
   summaries/           # Per-PMCID summary outputs
   outputs/             # Full combined per-PMCID outputs (all stages merged)
   eval_results/        # Evaluation results (variant scores, sentence scores, aggregate metrics)
```

**Changes:**

1. **Replace the current flat per-PMCID output** (one `{PMCID}.json` per article in the run directory) with the structured layout above.

2. **Save `config.yaml` at run start** — copy the resolved config into the run directory so the exact parameters are always co-located with results.

3. **Save `metadata.yaml` at run start, update at end** — includes timestamp, list of PMCIDs, stages requested, git commit hash, and a completion status updated when the run finishes.

4. **Stage-specific directories** (`sentences/`, `citations/`, `summaries/`) each contain per-PMCID JSON files using the experiment Pydantic models. This mirrors how the experiment `run.py` scripts already save outputs.

5. **`variants.json`** is a single file (not a directory) since variant extraction is fast and the output is compact — a dict mapping `pmcid -> list[str]`.

6. **`outputs/`** contains the full merged result per PMCID (all stages combined), equivalent to what the current pipeline produces.

7. **`eval_results/`** is populated either by `--eval` mode or by the eval pipeline, containing per-PMCID eval JSON files plus an `aggregate.json` with overall metrics.

---

## Summary of deletions

| File | Lines deleted | Replaced by |
|------|-------------|-------------|
| `pipeline.py` — `extract_variants()` | ~50 | `VariantExtractor.get_variants()` |
| `pipeline.py` — `parse_sentence_output()` | ~40 | `sentence_generation/utils.py` |
| `pipeline.py` — `generate_sentences()` | ~75 | `SentenceGenerator.generate()` |
| `pipeline.py` — `parse_citation_output()` | ~60 | `citations/utils.py` |
| `pipeline.py` — `find_citations()` | ~80 | `CitationFinder.find()` |
| `pipeline.py` — `generate_summary()` + formatters | ~70 | `SummaryGenerator.generate()` |
| `pipeline.py` — `initialize_variant_extraction()` | ~12 | Factory handles internally |
| `eval_pipeline.py` — `calculate_f1_metrics()` | ~15 | `benchmark_v2` |
| `eval_pipeline.py` — `evaluate_variants_for_pmcid()` | ~65 | `variant_finding/eval.py` |
| `eval_pipeline.py` — `evaluate_sentences_for_pmcid()` | ~120 | `sentence_generation/eval.py` |
| `eval_pipeline.py` — 6 custom dataclasses | ~85 | Experiment Pydantic models |

**Net reduction:** ~1,730 lines across both files -> ~550 lines total.

---

## Order of operations

1. **Phase 1** first — this is the highest-value change and can be done incrementally (one stage at a time).
2. **Phase 2** next — aligning data models makes Phase 4 straightforward.
3. **Phase 3** alongside Phase 1 — updating config is a natural companion to wiring in factories.
4. **Phase 6** alongside Phase 2 — the new output structure is easiest to implement while reworking data models.
5. **Phase 4** after Phases 1-2-6 — eval consolidation depends on aligned output models and the `eval_results/` directory.
6. **Phase 5** last — convenience feature, not blocking.

Each phase can be tested independently by running the pipeline on a single PMCID and comparing outputs to the previous version.
