"""
PGxMine variant extraction methodology experiments.

Implements three experiments to test PGxMine's core innovations:

1. pgxmine_context_aware: Context-aware star allele detection
   - Uses PubTator to identify Gene entities
   - Applies star allele regex ONLY after gene mentions
   - Tests if narrow context improves precision

2. pgxmine_normalized: Comprehensive normalization
   - Broad variant extraction with regex
   - Applies PGxMine's 157-pattern normalization
   - Tests if normalization compensates for messier extraction

3. pgxmine_full: Complete PGxMine pipeline
   - Sentence-level filtering (Chemical AND Mutation/Gene co-occurrence)
   - Context-aware extraction + normalization
   - Tests end-to-end methodology

References:
- PGxMine star allele detection: pgxmine/findPGxSentences.py:33
- PGxMine normalization: pgxmine/utils/__init__.py:11-235
"""

import json
import re
import time

import requests
from loguru import logger

from src.modules.variant_finding.utils import get_combined_text
from src.modules.variant_finding.pgxmine_normalization import normalize_mutation
from src.utils import ROOT

PUBTATOR_API_URL = (
    "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocjson"
)
REQUEST_DELAY = 0.35

_pmid_mapping = None
_last_request_time = 0.0


# ============================================================================
# PubTator Integration
# ============================================================================


def _get_pmid_mapping() -> dict[str, str]:
    """Get or initialize the PMCID-to-PMID mapping singleton."""
    global _pmid_mapping
    if _pmid_mapping is None:
        data_path = ROOT / "data" / "benchmark_v2" / "variant_bench.jsonl"
        _pmid_mapping = {}
        with open(data_path) as f:
            for line in f:
                record = json.loads(line)
                _pmid_mapping[record["pmcid"]] = record["pmid"]
    return _pmid_mapping


def _fetch_pubtator_annotations(pmid: str, full_text: bool = True) -> dict | None:
    """Fetch annotations from PubTator3 API for a given PMID."""
    global _last_request_time

    # Rate limiting
    elapsed = time.time() - _last_request_time
    if elapsed < REQUEST_DELAY:
        time.sleep(REQUEST_DELAY - elapsed)

    params = {"pmids": pmid}
    if full_text:
        params["full"] = "true"

    try:
        response = requests.get(PUBTATOR_API_URL, params=params, timeout=30)
        response.raise_for_status()
        _last_request_time = time.time()
        return response.json()
    except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
        logger.error(f"Failed to fetch PubTator annotations for PMID {pmid}: {e}")
        _last_request_time = time.time()
        return None


def _extract_entities_from_biocjson(
    biocjson: dict, entity_types: list[str]
) -> list[dict]:
    """Extract entities of specified types from BioC JSON response.

    Args:
        biocjson: PubTator BioC JSON response
        entity_types: List of entity types to extract (e.g., ["Gene", "Chemical"])

    Returns:
        List of entity dicts with keys: text, type, start, end, passage_offset
    """
    entities = []

    documents = []
    if isinstance(biocjson, dict):
        if "PubTator3" in biocjson:
            documents = biocjson.get("PubTator3", [])
        else:
            documents = [biocjson]
    elif isinstance(biocjson, list):
        documents = biocjson

    for doc in documents:
        for passage in doc.get("passages", []):
            passage_offset = passage.get("offset", 0)

            for annotation in passage.get("annotations", []):
                infons = annotation.get("infons", {})
                ann_type = infons.get("type", "")

                if ann_type in entity_types:
                    # Get text and location
                    text = annotation.get("text", "").strip()
                    locations = annotation.get("locations", [])

                    if text and locations:
                        for loc in locations:
                            entities.append({
                                "text": text,
                                "type": ann_type,
                                "start": loc.get("offset", 0),
                                "end": loc.get("offset", 0) + loc.get("length", 0),
                                "passage_offset": passage_offset,
                            })

    return entities


# ============================================================================
# Text Processing
# ============================================================================


def _split_into_sentences(text: str) -> list[dict]:
    """Split text into sentences with character offsets.

    Uses simple sentence splitting (periods, exclamation marks, question marks).

    Returns:
        List of dicts with keys: text, start, end
    """
    sentences = []
    # Simple sentence boundary detection
    pattern = r'([.!?]+[\s\n]+)'
    parts = re.split(pattern, text)

    offset = 0
    current_sentence = ""
    current_start = 0

    for i, part in enumerate(parts):
        if re.match(pattern, part):
            # End of sentence
            current_sentence += part
            sentences.append({
                "text": current_sentence,
                "start": current_start,
                "end": offset + len(part),
            })
            offset += len(part)
            current_sentence = ""
            current_start = offset
        else:
            # Sentence content
            current_sentence += part
            offset += len(part)

    # Add remaining text as final sentence
    if current_sentence.strip():
        sentences.append({
            "text": current_sentence,
            "start": current_start,
            "end": offset,
        })

    return sentences


def _filter_sentences_with_chem_variant(
    sentences: list[dict], gene_entities: list[dict], chem_entities: list[dict],
    mutation_entities: list[dict]
) -> list[dict]:
    """Filter to sentences containing both Chemical AND (Mutation OR Gene).

    This implements PGxMine's sentence-level filtering strategy.

    Args:
        sentences: List of sentence dicts with start/end offsets
        gene_entities: List of Gene entity dicts
        chem_entities: List of Chemical entity dicts
        mutation_entities: List of Mutation entity dicts

    Returns:
        Filtered list of sentences that meet the criteria
    """
    filtered = []

    for sent in sentences:
        sent_start = sent["start"]
        sent_end = sent["end"]

        # Check for Chemical entity in this sentence
        has_chemical = any(
            ent["start"] >= sent_start and ent["end"] <= sent_end
            for ent in chem_entities
        )

        # Check for Mutation or Gene entity in this sentence
        has_variant = any(
            ent["start"] >= sent_start and ent["end"] <= sent_end
            for ent in mutation_entities + gene_entities
        )

        if has_chemical and has_variant:
            filtered.append(sent)

    return filtered


# ============================================================================
# Variant Extraction
# ============================================================================


def _extract_star_alleles_after_genes(
    text: str, gene_entities: list[dict], context_window: int = 50
) -> set[str]:
    """Extract star alleles using context-aware detection.

    Applies PGxMine's star allele regex ONLY after gene mentions.

    Args:
        text: Full article text
        gene_entities: List of Gene entity dicts with start/end positions
        context_window: Characters after gene to search for star alleles

    Returns:
        Set of normalized star alleles (e.g., "CYP2D6*4")
    """
    star_alleles = set()

    # PGxMine's exact star allele regex from findPGxSentences.py:33
    regex = r'^(,|and|or|/|\s|\+)*(?P<main>\*\s*[0-9]([\w:]*\w+)?)'

    for gene_ent in gene_entities:
        gene_name = gene_ent["text"].upper()
        gene_end = gene_ent["end"]

        # Search in window after gene mention
        search_start = gene_end
        search_end = min(gene_end + context_window, len(text))
        window_text = text[search_start:search_end]

        # Find star alleles in this window
        offset = 0
        while offset < len(window_text):
            match = re.search(regex, window_text[offset:])
            if not match:
                break

            _, length = match.span()
            start_pos, end_pos = match.span('main')
            allele_text = match.group('main')

            # Extract allele number (everything after the *)
            allele_num = allele_text.strip()[1:].strip()

            # Format as GENE*ALLELE
            if allele_num:
                star_alleles.add(f"{gene_name}*{allele_num}")

            offset += length

    return star_alleles


def _extract_rsids(text: str) -> set[str]:
    """Extract rsID variants from text."""
    pattern = r'\brs\d{4,}\b'
    matches = re.findall(pattern, text, re.IGNORECASE)
    return {m.lower() for m in matches}


def _extract_broad_variants(text: str) -> set[str]:
    """Extract variants using broad regex patterns.

    Returns raw, unnormalized variants for testing normalization impact.
    """
    variants = set()

    # Star alleles (anywhere in text)
    star_pattern = r'\*\s*[0-9][\w:]*'
    matches = re.findall(star_pattern, text)
    variants.update(matches)

    # rsIDs
    variants.update(_extract_rsids(text))

    # HLA alleles (basic pattern)
    hla_pattern = r'\b(?:HLA-)?([ABC]|DRB[1345]|DQ[AB]1|DP[AB]1)\*\d{2,}:?\d{0,2}\b'
    matches = re.findall(hla_pattern, text, re.IGNORECASE)
    # HLA matches return just the gene, so we need to extract full match
    for match in re.finditer(hla_pattern, text, re.IGNORECASE):
        variants.add(match.group(0))

    return variants


# ============================================================================
# Experiment Implementations
# ============================================================================


def pgxmine_context_aware_extract(pmcid: str) -> list[str]:
    """Experiment 1: Context-aware star allele detection.

    Tests PGxMine's core innovation: detecting star alleles only after genes.

    Methodology:
    1. Get article text
    2. Use PubTator to identify Gene entities
    3. Apply star allele regex ONLY after gene mentions (50 char window)
    4. Extract rsIDs globally
    5. Return unique variants

    Expected Insight: Does context-aware detection improve precision vs broad extraction?
    """
    # Get PMID mapping
    pmid_mapping = _get_pmid_mapping()
    pmid = pmid_mapping.get(pmcid)
    if not pmid:
        logger.warning(f"No PMID found for {pmcid}")
        return []

    # Get article text
    text, _ = get_combined_text(pmcid)

    # Get Gene entities from PubTator
    biocjson = _fetch_pubtator_annotations(pmid, full_text=True)
    if biocjson is None:
        logger.warning(f"No PubTator data for {pmcid} (PMID {pmid})")
        return []

    gene_entities = _extract_entities_from_biocjson(biocjson, ["Gene"])

    if not gene_entities:
        logger.info(f"No gene entities found for {pmcid}")
        return []

    # Extract star alleles using context-aware method
    star_alleles = _extract_star_alleles_after_genes(text, gene_entities)

    # Extract rsIDs globally
    rsids = _extract_rsids(text)

    # Combine and return
    variants = star_alleles | rsids
    logger.info(
        f"Context-aware extraction: {len(variants)} variants for {pmcid} "
        f"({len(star_alleles)} star alleles, {len(rsids)} rsIDs)"
    )

    return list(variants)


def pgxmine_normalized_extract(pmcid: str) -> list[str]:
    """Experiment 2: Comprehensive normalization impact.

    Tests whether PGxMine's 157-pattern normalization compensates for messier extraction.

    Methodology:
    1. Get article text
    2. Extract variants with broad patterns (star alleles, rsIDs, HLA)
    3. Apply PGxMine's normalization to each candidate
    4. Return normalized variants

    Expected Insight: Does normalization overcome broad, noisy extraction?
    """
    # Get article text
    text, _ = get_combined_text(pmcid)

    # Extract variants broadly
    raw_variants = _extract_broad_variants(text)

    # Normalize each variant
    normalized_variants = set()
    for variant in raw_variants:
        normalized = normalize_mutation(variant)
        if normalized:
            normalized_variants.add(normalized)
        else:
            # If normalization fails, keep original (for rsIDs and HLA)
            normalized_variants.add(variant)

    logger.info(
        f"Normalized extraction: {len(normalized_variants)} variants for {pmcid} "
        f"({len(raw_variants)} raw -> {len(normalized_variants)} normalized)"
    )

    return list(normalized_variants)


def pgxmine_full_extract(pmcid: str) -> list[str]:
    """Experiment 3: Complete PGxMine pipeline.

    Tests the full PGxMine methodology end-to-end.

    Methodology:
    1. Get article text, split into sentences
    2. Get PubTator annotations for Genes, Chemicals, Mutations
    3. Filter to sentences with BOTH Chemical AND (Mutation OR Gene)
    4. Extract star alleles (context-aware) + rsIDs from filtered sentences
    5. Apply normalization
    6. Return unique variants

    Expected Insight: How does complete pipeline compare to regex_v5 baseline?
    """
    # Get PMID mapping
    pmid_mapping = _get_pmid_mapping()
    pmid = pmid_mapping.get(pmcid)
    if not pmid:
        logger.warning(f"No PMID found for {pmcid}")
        return []

    # Get article text
    text, _ = get_combined_text(pmcid)

    # Get PubTator annotations
    biocjson = _fetch_pubtator_annotations(pmid, full_text=True)
    if biocjson is None:
        logger.warning(f"No PubTator data for {pmcid} (PMID {pmid})")
        return []

    gene_entities = _extract_entities_from_biocjson(biocjson, ["Gene"])
    chem_entities = _extract_entities_from_biocjson(biocjson, ["Chemical"])
    mutation_entities = _extract_entities_from_biocjson(
        biocjson, ["Mutation", "SNP", "DNAMutation", "ProteinMutation"]
    )

    logger.info(
        f"Entities for {pmcid}: {len(gene_entities)} genes, "
        f"{len(chem_entities)} chemicals, {len(mutation_entities)} mutations"
    )

    # Split into sentences
    sentences = _split_into_sentences(text)

    # Filter to relevant sentences
    filtered_sentences = _filter_sentences_with_chem_variant(
        sentences, gene_entities, chem_entities, mutation_entities
    )

    logger.info(
        f"Sentence filtering: {len(filtered_sentences)}/{len(sentences)} sentences "
        f"contain both Chemical and Variant entities"
    )

    if not filtered_sentences:
        logger.info(f"No relevant sentences found for {pmcid}")
        return []

    # Combine filtered sentence text
    filtered_text = " ".join(sent["text"] for sent in filtered_sentences)

    # Extract variants from filtered text

    # Star alleles using context-aware detection
    # Filter gene entities to those in filtered sentences
    filtered_gene_entities = [
        ent for ent in gene_entities
        if any(
            sent["start"] <= ent["start"] and ent["end"] <= sent["end"]
            for sent in filtered_sentences
        )
    ]

    star_alleles = _extract_star_alleles_after_genes(
        text, filtered_gene_entities
    )

    # rsIDs from filtered text
    rsids = _extract_rsids(filtered_text)

    # Combine all variants
    raw_variants = star_alleles | rsids

    # Apply normalization
    normalized_variants = set()
    for variant in raw_variants:
        normalized = normalize_mutation(variant)
        if normalized:
            normalized_variants.add(normalized)
        else:
            normalized_variants.add(variant)

    logger.info(
        f"Full pipeline: {len(normalized_variants)} variants for {pmcid} "
        f"({len(star_alleles)} star alleles, {len(rsids)} rsIDs)"
    )

    return list(normalized_variants)
