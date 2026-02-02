# Codebase Refactoring Analysis

## Critical Issues

### 1. Massive Code Duplication — Variant Extraction (5 versions)

`src/experiments/variant_finding/regex_variants/` has `extract_variants.py` through `extract_variants_v5.py` — ~1,700 lines of nearly identical code. Functions like `extract_rsids()`, `extract_star_alleles()`, and `extract_hla_alleles()` are copy-pasted across all 5 files with minor tweaks.

**Fix:** Consolidate into a single parameterized module. Version differences should be config, not code.

### 2. Hardcoded Values Everywhere

- **Pharmacogene lists** hardcoded in multiple places with *different* lists (`extract_variants_v5.py:140-166` has 24 genes, `snp_expansion.py:55-70` has 12)
- **Magic numbers:** `min_distance = 800` (`extract_variants_v5.py:204`), similarity threshold `0.85` (`drug_benchmark.py:287`), penalty rate `0.05` (`drug_benchmark.py:382`)
- **Benchmark weights** hardcoded inline (`run_benchmark.py:146-151`)
- **Model names** hardcoded (`raw_sentence_ask.py:368`)
- **Tolerance levels** embedded in logic (`shared_utils.py:106-107`)

**Fix:** Create a central config file (YAML/JSON) for gene lists, thresholds, weights, and model names.

### 3. Monolithic Files

| File | Lines | Responsibility |
|------|-------|---------------|
| `pipeline.py` | 909 | Variant extraction, sentence generation, citations, summary |
| `eval_pipeline.py` | 824 | Full evaluation pipeline |
| `drug_benchmark.py` | 445 | Data transformation, matching, scoring, parsing |

Functions like `align_by_variant()` in `drug_benchmark.py:86-185` are 100+ lines with 3 nesting levels.

**Fix:** Split into smaller modules with single responsibilities (e.g., `pipeline/stages/`).

### 4. Global Singleton Anti-Pattern

Multiple files use mutable global state:

- `snp_expansion.py:23-24`
- `shared_utils.py:10-18`
- `extract_variants_v5.py:24`

```python
_snp_expander = None
def get_snp_expander():
    global _snp_expander
    if _snp_expander is None:
        _snp_expander = SNPExpander()
    return _snp_expander
```

**Fix:** Use dependency injection — pass models/expanders as function parameters.

### 5. Duplicated Alignment Logic

`drug_benchmark.py:86-185` and `pheno_benchmark.py:34-186` have nearly identical ~100-line `align_by_variant()` functions (expand annotations, build index, match by priority, track unmatched).

**Fix:** Extract into a shared `alignment.py` utility module.

---

## Moderate Issues

### 6. Inconsistent Error Handling

No consistent contract across the codebase:

| Location | Strategy |
|----------|----------|
| `utils.py:27-28` | Returns `""` on `FileNotFoundError` |
| `snp_expansion.py:150-152` | Returns `[]` on error |
| `utils.py:210-212` | Re-raises exception |
| `shared_utils.py:50-51` | Broad `except Exception` with silent fallback |
| `utils_bioc.py:49-50` | `except (json.JSONDecodeError, KeyError): pass` — silent failure |

**Fix:** Define a consistent error handling strategy. Create custom exception types. Avoid bare `except Exception`.

### 7. Inconsistent Path Construction

39 files construct root paths using fragile chains like `Path(__file__).parent.parent.parent`.

**Fix:** Create a single `src/paths.py` with a `ROOT` constant, import everywhere.

### 8. Mixed Abstraction Levels in Utils

`src/utils.py` combines file I/O, text extraction, section parsing, model normalization, and LLM calls in one file.

**Fix:** Split into `file_utils.py`, `text_utils.py`, `llm_utils.py`.

### 9. Long Parameter Lists

Functions like `run_experiment()` in `raw_sentence_ask.py:165-172` take 6+ parameters.

**Fix:** Use config dataclasses to group related parameters.

### 10. No Tests

No unit or integration tests found. This makes refactoring risky since there's no safety net to catch regressions.

**Fix:** Add tests before refactoring, starting with the most critical shared utilities.

---

## What's Done Well

- Recent consolidation of LLM calls into `src/utils.py` (good direction)
- Good use of dataclasses in newer code (`VariantBenchResult`)
- Reasonable type hints in newer files
- Modular experiment structure

---

## Suggested Refactoring Order

1. **Consolidate variant extraction** into one parameterized module (biggest win — eliminates ~1,400 duplicate lines)
2. **Extract config** — move gene lists, thresholds, weights to a config file
3. **Share alignment logic** between drug/pheno benchmarks
4. **Add `src/paths.py`** with a single `ROOT` constant
5. **Split `pipeline.py`** into separate stage modules
6. **Replace globals** with dependency injection
7. **Standardize error handling** across the codebase
8. **Add tests** for shared utilities before further refactoring
