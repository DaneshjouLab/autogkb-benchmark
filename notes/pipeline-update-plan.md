# Pipeline Update Plan (March 2026)

Based on curator feedback from Feb 2026 review. See `feb-feedback.md` for raw feedback.

## Change 1: Remove Clinical Implications from Summary (Feedback #1)
**File:** `generation/modules/summary/methods/prompts/basic_summary.yaml`

- In **v1**, remove the `## Clinical Implications` section from the output format and instructions. The summary should only produce `## Background` and `## Key Findings`.
- Create a new prompt version **v3** (leaving v1/v2 intact for reproducibility) with the updated format.
- Update `generation/configs/base_config.yaml` to point to `prompt_version: v3`.

## Change 2: Ground sentence generation in Results & Methods only (Feedback #2, #5)
**File:** `shared/utils.py`

- Create a new function `get_methods_and_results_text(pmcid)` that extracts only **Methods** and **Results** sections (drop Discussion/Conclusions from `get_methods_and_conclusions_text`).
- Keep the existing function for backward compatibility.

**File:** `generation/modules/sentence_generation/methods/batch_judge_ask.py`
- Switch from `get_methods_and_conclusions_text` to the new `get_methods_and_results_text` so the LLM only sees Methods and Results.

**File:** `generation/modules/sentence_generation/methods/prompts/batch_judge_ask.yaml`
- Create a new prompt version **v6** based on v5 with additional instructions:
  - "Only report genotype comparisons and statistical values that are **explicitly stated** in the article text provided. Do not infer or construct comparisons not present in the source."
  - "If a p-value or statistical measure is present in the text, you MUST include it. Never state that statistical values were 'not provided' if they appear in the article."
- Update `generation/configs/base_config.yaml` to `prompt_version: v6`.

## Change 3: Exhaustive multi-association capture (Feedback #3, #6, #7)

The underlying problem: the Variant Association section misses findings that Key Findings captures, and groups distinct associations together. Both are LLM-generated — the gap is that the sentence generation prompt is too narrow in what it asks for, not that it lacks information.

**Approach: prompt-only fix (no pipeline reorder)**

**File:** `generation/modules/sentence_generation/methods/prompts/batch_judge_ask.yaml` (v6)
- Add explicit instruction: "Be **exhaustive** — capture ALL pharmacogenomic associations reported in the article for each variant, including: positive associations, negative associations ('no association' findings), PK effects, efficacy outcomes, toxicity/adverse event outcomes, and dosing implications."
- Add: "Do not limit yourself to one association per variant. If the article reports multiple outcomes for the same variant, generate a separate VARIANT/SENTENCE/EXPLANATION block for each."
- Add: "If a single sentence in the article reports different effect directions for different phenotypes (e.g., decreased risk for X but increased risk for Y), split into separate entries — one per phenotype-direction pair."

**File:** `generation/modules/sentence_generation/utils.py`
- Verify `parse_batch_output` correctly appends rather than overwrites when the same variant appears in multiple VARIANT/SENTENCE/EXPLANATION blocks (it already returns `dict[str, list[GeneratedSentence]]` so likely fine, but confirm).

**Fallback plan:** If prompt-only improvements don't close the gap after testing on the reviewer papers (PMC12331468, PMC2859392, PMC4916189, PMC6465603, PMC11971672), revisit the two-pass pipeline architecture.

## Change 4: Filter variants to only those actually studied (Feedback #4, #12)
**File:** `generation/modules/variant_finding/utils.py` or new file `generation/modules/variant_finding/methods/regex_v5_filtered.py`

- After extracting all variants from the full text (including supplements), do a **second pass** to check which variants appear in the Methods and Results sections specifically.
- Tag each variant with where it was found: `methods`, `results`, `discussion`, `supplement`.
- Only pass forward variants found in Methods or Results (or supplements). Variants mentioned only in Discussion/Introduction are excluded.
- This also resolves the "was not mentioned in the article" text issue (Feedback #12) — by filtering non-studied variants upstream, the LLM no longer needs to detect them via fallback text, so the incorrect phrasing problem goes away.

## Change 5: Resolve star allele / rsID aliasing (Feedback #8, #9)
**File:** `shared/term_normalization/variant_search.py` or new file `generation/modules/variant_finding/alias_resolver.py`

- Build or integrate an alias mapping: star allele <-> rsID (e.g., DPYD*9 = rs1801265). PharmGKB's API already provides some of this data.
- After variant extraction, group aliases together so that if both `DPYD*9` and `rs1801265` are found, they're treated as the same variant.
- When generating sentences, if the findings are about **metabolizer phenotypes** (PM, IM, NM, etc.), keep the association sentence identical across all constituent star alleles in that group.

**File:** `generation/modules/sentence_generation/methods/prompts/batch_judge_ask.yaml` (v6)
- Add: "If multiple variants belong to the same metabolizer phenotype group and the paper's findings are about the phenotype (e.g., poor metabolizers), use the **same association sentence** for each constituent allele."

## Change 6: Stop using "null" for empty genotypes (Feedback #10)
**File:** `generation/modules/sentence_generation/methods/prompts/batch_judge_ask.yaml` (v6)

- Add instruction: "Never use the word 'null' to describe empty or unknown genotype fields. Use an empty string or omit the field. The term 'null' has a specific meaning in pharmacogenomics (gene deletion alleles)."

Also grep the codebase for any hardcoded "null" defaults in models or output formatting and replace them.

## Change 7: Citation grounding — prioritize Results & Methods (Feedback #2)
**File:** `generation/modules/citations/methods/prompts/one_shot_citations.yaml`

- Create **v3** with instruction: "Prioritize citing sentences from the Methods and Results sections. Avoid citing from the Discussion or Introduction unless the relevant data (e.g., statistical values, tables) only appears there."

Note: we keep the full article text available for citation search (via `get_markdown_text`) rather than restricting to Methods+Results only. Some key statistical results appear in tables or supplementary sections that sit outside those markdown headers. The prompt instruction is sufficient to steer citations away from Discussion-section speculation.

## Change 8: Plus-strand allele reporting (Feedback #11)
This requires a reference database of gene strand orientation. The LLM cannot reliably determine which genes are on the minus strand without external data, so a prompt-only fix won't work.

- Integrate a strand lookup (e.g., from NCBI Gene or Ensembl) into the term normalization stage to programmatically complement alleles for minus-strand genes.
- Separate work item — requires research into available strand orientation APIs/databases.

## Change 9: Fix analysis/scoring system (Feedback #13)
This is in the **benchmark** code, not the generation pipeline. The scoring system is in `benchmark/v1/` and `benchmark/v2/`.

- The current false positive logic in the benchmark incorrectly labels valid annotations as false positives when they appear in both Drug and Phenotype annotation categories.
- Recommended as a **separate work item** since it doesn't affect generation quality.

## Summary of files to modify

| File | Changes |
|------|---------|
| `shared/utils.py` | Add `get_methods_and_results_text()` |
| `generation/configs/base_config.yaml` | Update prompt versions (v6 sentence, v3 summary, v3 citation) |
| `generation/modules/variant_finding/utils.py` | Add section-aware variant filtering |
| `generation/modules/sentence_generation/methods/batch_judge_ask.py` | Use new text extraction function |
| `generation/modules/sentence_generation/methods/prompts/batch_judge_ask.yaml` | New v6 prompt (Changes 2, 3, 5, 6) |
| `generation/modules/sentence_generation/utils.py` | Verify multi-entry parsing for same variant |
| `generation/modules/citations/methods/prompts/one_shot_citations.yaml` | New v3 prompt (prioritize Results & Methods) |
| `generation/modules/summary/methods/prompts/basic_summary.yaml` | New v3 prompt (no Clinical Implications) |

## Suggested implementation order
1. Changes 1, 3, 6, 7 (prompt-only, low risk — v6 sentence prompt, v3 summary prompt, v3 citation prompt)
2. Change 2 (text extraction scoping — new util function + wiring in batch_judge_ask.py)
3. Change 4 (variant filtering)
4. Change 5 (star allele / rsID aliasing)
5. Changes 8, 9 (longer-term / separate workstreams)

## Validation
After implementing steps 1-4, re-run the pipeline on the five papers reviewed by curators and compare output against their feedback:
- PMC12331468 (Caroline)
- PMC2859392 (Li)
- PMC4916189 (Li)
- PMC6465603 (Evangelia)
- PMC11971672 (Katrin)
