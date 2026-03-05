# Refactor (March 4, 2026)

## Goals
- Separate benchmark and generation pipeline into independent root-level packages
- Both interact with the `data/` folder without needing to know about each other
- Single command runs the entire pipeline against a PMID (converts to PMCID, generates annotations)
- Generation outputs saved to `data/generations.jsonl`

## What Changed

### Package Structure
Moved from a monolithic `autogkb_pipeline/` package (formerly `src/`) to four independent root-level packages:

| Package | Purpose |
|---|---|
| `generation/` | Pipeline stages: variant finding, term normalization, sentence generation, citations, summary |
| `benchmark/` | Evaluation: v1 (drug/pheno/FA/study params) and v2 (modular per-stage evals) |
| `shared/` | Utilities (LLM calls, paths), data setup, and term normalization lookups |
| `pubmed_downloader/` | Article downloading (PMID→PMCID→HTML→markdown) |

### Key Details
- `pyproject.toml` uses hatchling, packages `generation`, `benchmark`, `shared`
- Entry point: `generate` → `generation.pipeline:main`
- `pixi.toml` defines tasks: `generate`, `benchmark`, `variant-setup`, `variants`, `fields`, `setup-repo`
- All imports updated from `autogkb_pipeline.*` → `generation.*` / `benchmark.*` / `shared.*`
- Old `autogkb_pipeline/` outputs, cached data, and deprecated code removed (~34k lines deleted)
- `generation/models.py` defines `GenerationRecord` and `GenerationMetadata` (Pydantic models)
- `generation/__main__.py` added for `python -m generation` support

### Removed
- `src/` — empty skeleton left over from original layout (only had `__pycache__`)
- `autogkb_pipeline/` — deprecated wrapper with just a warning `__init__.py`
- `scratch/` — one-off experiment scripts and old benchmark result JSONs
- Old generation outputs, eval results, and cached BioC/SNP data that lived inside the package
