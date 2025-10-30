from typing import Dict, List, Any, Optional
from difflib import SequenceMatcher
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from src.fa_benchmark.model_cache import ModelCache


def parse_variant_list(variants_text: Optional[str]) -> List[str]:
    """Split multi-variant strings into a list of tokens using common separators."""
    if not variants_text:
        return []
    # Reuse the same splitting logic used inside variant_coverage
    tokens = re.split(r'[,;|\s]+(?:\+\s*)?', variants_text)
    return [t.strip() for t in tokens if t and t.strip()]


def normalize_variant(variant: str) -> str:
    """Normalize variant tokens: lowercase rsIDs, standardize spacing/case for star alleles."""
    v = variant.strip()
    if v.lower().startswith('rs'):
        return v.lower()
    # For star alleles, keep case but normalize internal spaces
    return re.sub(r'\s+', '', v)


def expand_annotations_by_variant(annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Expand annotations with multiple Variant/Haplotypes into per-variant entries.
    Keeps all other fields identical and marks expanded entries internally.
    """
    expanded: List[Dict[str, Any]] = []
    for ann in annotations:
        variants_field = ann.get('Variant/Haplotypes')
        tokens = parse_variant_list(variants_field)
        if len(tokens) <= 1:
            expanded.append(ann)
            continue
        for tok in tokens:
            new_ann = dict(ann)
            new_ann['Variant/Haplotypes'] = normalize_variant(tok)
            new_ann['_expanded_from_multi_variant'] = True
            expanded.append(new_ann)
    return expanded

def validate_external_data(annotation: Dict[str, Any]) -> List[str]:
    """Validate external data format and nomenclature."""
    issues = []
    
    # Regex patterns for validation
    rsid_pattern = re.compile(r'^rs\d+$', re.IGNORECASE)
    star_allele_pattern = re.compile(r'^[A-Z0-9]+\*\d+$')
    gene_pattern = re.compile(r'^[A-Z0-9]+$')
    
    # Validate variant nomenclature
    variants = annotation.get('Variant/Haplotypes', '')
    if variants:
        variant_list = re.split(r'[,;|\s]+(?:\+\s*)?', variants)
        for variant in variant_list:
            variant = variant.strip()
            if not variant:
                continue
                
            # Check rsID format
            if variant.lower().startswith('rs'):
                if not rsid_pattern.match(variant):
                    issues.append(f"Invalid rsID format: {variant}")
            
            # Check star allele format
            elif '*' in variant:
                if not star_allele_pattern.match(variant):
                    issues.append(f"Invalid star allele format: {variant}")
    
    # Validate gene names
    gene = annotation.get('Gene', '')
    if gene and not gene_pattern.match(gene):
        issues.append(f"Invalid gene name format: {gene}")
    
    return issues


def validate_field_dependencies(annotation: Dict[str, Any]) -> List[str]:
    """Validate logical dependencies between fields."""
    issues = []
    
    # If "Gene/gene product" is filled, "Gene" must be filled
    gene_product = annotation.get('Gene/gene product')
    gene = annotation.get('Gene')
    if gene_product and not gene:
        issues.append("Gene field required when Gene/gene product is specified")
    
    # If "Comparison Allele(s)" exists, "Variant/Haplotypes" must exist
    comparison_alleles = annotation.get('Comparison Allele(s) or Genotype(s)')
    variants = annotation.get('Variant/Haplotypes')
    if comparison_alleles and not variants:
        issues.append("Variant/Haplotypes required when Comparison Allele(s) is specified")
    
    # If "Direction of effect" is set, "Is/Is Not associated" should be "Associated with"
    direction = annotation.get('Direction of effect')
    association = annotation.get('Is/Is Not associated')
    if direction and association != 'Associated with':
        issues.append("Direction of effect requires 'Associated with' status")
    
    # If "Functional terms" is specified, "Gene/gene product" should be filled
    functional_terms = annotation.get('Functional terms')
    gene_product = annotation.get('Gene/gene product')
    if functional_terms and not gene_product:
        issues.append("Gene/gene product recommended when Functional terms is specified")
    
    return issues


def validate_annotation_consistency(annotation: Dict[str, Any], 
                                 study_params: List[Dict[str, Any]]) -> List[str]:
    """Check annotation consistency with study parameters."""
    issues = []
    
    # Check Variant Annotation ID exists in study_parameters
    variant_id = annotation.get('Variant Annotation ID')
    if variant_id:
        found_in_study = any(
            sp.get('Variant Annotation ID') == variant_id 
            for sp in study_params
        )
        if not found_in_study:
            issues.append(f"Variant Annotation ID {variant_id} not found in study_parameters")
    
    # Check PMID consistency
    annotation_pmid = annotation.get('PMID')
    if annotation_pmid and study_params:
        study_pmids = {sp.get('PMID') for sp in study_params if sp.get('PMID')}
        if study_pmids and annotation_pmid not in study_pmids:
            issues.append(f"PMID {annotation_pmid} inconsistent with study parameters")
    
    return issues


def validate_all_dependencies(annotation: Dict[str, Any], 
                            study_params: Optional[List[Dict[str, Any]]] = None) -> List[str]:
    """Run all validation checks on an annotation."""
    issues = []
    
    # Run all validation methods
    issues.extend(validate_external_data(annotation))
    issues.extend(validate_field_dependencies(annotation))
    
    if study_params:
        issues.extend(validate_annotation_consistency(annotation, study_params))
    
    return issues


def evaluate_functional_analysis(ground_truth: List[Dict[str, Any]], 
                                predictions: List[Dict[str, Any]],
                                study_parameters: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Evaluate LLM predictions against ground truth for functional analysis annotations.
    
    Args:
        ground_truth: List of ground truth annotation dictionaries
        predictions: List of predicted annotation dictionaries
        
    Returns:
        Dictionary containing overall score and field-specific scores
    """
    
    if len(ground_truth) != len(predictions):
        raise ValueError(f"Length mismatch: GT={len(ground_truth)}, Pred={len(predictions)}")
    
    # PubMedBERT model init for semantic similarity
    model = ModelCache.get_model()
    
    # Field eval functions
    def exact_match(gt_val: Any, pred_val: Any) -> float:
        """1:1 exact match"""
        if gt_val is None and pred_val is None:
            return 1.0
        if gt_val is None or pred_val is None:
            return 0.0
        return 1.0 if str(gt_val).strip().lower() == str(pred_val).strip().lower() else 0.0
    
    def semantic_similarity(gt_val: Any, pred_val: Any) -> float:
        """Semantic similarity using PubMedBERT embeddings"""
        if gt_val is None and pred_val is None:
            return 1.0
        if gt_val is None or pred_val is None:
            return 0.0
        
        gt_str = str(gt_val).strip()
        pred_str = str(pred_val).strip()
        
        if gt_str == pred_str:
            return 1.0
        
        try:
            # embedding based similarity
            embeddings = model.encode([gt_str, pred_str])
            gt_embedding = embeddings[0]
            pred_embedding = embeddings[1]
            similarity = np.dot(gt_embedding, pred_embedding) / (
                np.linalg.norm(gt_embedding) * np.linalg.norm(pred_embedding)
            )
            return float(similarity)
        except Exception as e:
            print(f"Error in semantic similarity: {e}")
            # sequence matcher fallback
            return SequenceMatcher(None, gt_str.lower(), pred_str.lower()).ratio()
    
    def variant_coverage(gt_variants: str, pred_variants: str) -> float:
        """
        Evaluate variant coverage with proper parsing and wild-type handling.
        - Penalize for missing GT variants
        - Don't penalize for extra predicted variants (unless they're significant)
        - Exclude wild-type alleles from penalty
        - Use exact match for rsIDs and star alleles
        - Use semantic similarity for phenotype descriptions
        """
        if not gt_variants or not pred_variants:
            return 1.0 if not gt_variants and not pred_variants else 0.0
        
        # Helpers to check the type of variant (wild-type, rsID, star allele)
        def is_wildtype(variant: str) -> bool:
            variant_lower = variant.lower().strip()
            wildtype_indicators = ['wild type', 'wildtype', 'wt', '*1']
            return any(wt in variant_lower for wt in wildtype_indicators)
        
        def is_rsid(variant: str) -> bool:
            return variant.strip().lower().startswith('rs')
        
        def is_star_allele(variant: str) -> bool:
            return '*' in variant.strip()
        
        # Parse variants (handle multiple separators)
        import re
        gt_list = re.split(r'[,;|\s]+(?:\+\s*)?', gt_variants)
        pred_list = re.split(r'[,;|\s]+(?:\+\s*)?', pred_variants)
        
        gt_list = [v.strip() for v in gt_list if v.strip()]
        pred_list = [v.strip() for v in pred_list if v.strip()]
        
        # Filter out wild-type from ground truth (don't penalize for missing these)
        gt_list_filtered = [v for v in gt_list if not is_wildtype(v)]
        
        if not gt_list_filtered:
            # If all GT variants were wild-type, just check if prediction is reasonable
            return 1.0 if pred_list else 0.0
        
        # Calculate coverage
        covered_count = 0
        for gt_var in gt_list_filtered:
            gt_var_lower = gt_var.lower()
            
            # Exact match for rsIDs and star alleles
            if is_rsid(gt_var) or is_star_allele(gt_var):
                if any(gt_var_lower == pred_var.lower() for pred_var in pred_list):
                    covered_count += 1
            else:
                # Semantic/fuzzy match for phenotype descriptions
                best_match_score = 0.0
                for pred_var in pred_list:
                    # Check for substring matches
                    if gt_var_lower in pred_var.lower() or pred_var.lower() in gt_var_lower:
                        best_match_score = 1.0
                        break
                    # Use sequence matching for similarity
                    similarity = SequenceMatcher(None, gt_var_lower, pred_var.lower()).ratio()
                    best_match_score = max(best_match_score, similarity)
                
                # Consider covered if similarity > 0.8
                if best_match_score > 0.8:
                    covered_count += 1
        
        # Coverage score: proportion of GT variants found in predictions
        coverage = covered_count / len(gt_list_filtered)
        return coverage
    
    def category_match(gt_val: str, pred_val: str, valid_categories: List[str]) -> float:
        """Category classification accuracy"""
        if not gt_val and not pred_val:
            return 1.0
        if not gt_val or not pred_val:
            return 0.0
        
        gt_lower = gt_val.lower().strip()
        pred_lower = pred_val.lower().strip()
        
        return 1.0 if gt_lower == pred_lower else 0.0
    
    # Field definitions and evaluation methods
    field_evaluators = {
        'Variant/Haplotypes': variant_coverage,
        'Gene': semantic_similarity,
        'Drug(s)': semantic_similarity,
        'PMID': exact_match,
        'Phenotype Category': lambda gt, pred: category_match(gt, pred, 
            ['efficacy', 'toxicity', 'dosage', 'metabolism/pk', 'pd', 'other']),
        'Significance': lambda gt, pred: category_match(gt, pred, 
            ['yes', 'no', 'not stated']),
        'Alleles': semantic_similarity,
        'Specialty Population': semantic_similarity,
        'Assay type': semantic_similarity,
        'Metabolizer types': semantic_similarity,
        'isPlural': lambda gt, pred: category_match(gt, pred, ['is', 'are']),
        'Is/Is Not associated': lambda gt, pred: category_match(gt, pred, 
            ['associated with', 'not associated with']),
        'Direction of effect': lambda gt, pred: category_match(gt, pred, 
            ['increased', 'decreased']),
        'Functional terms': semantic_similarity,
        'Gene/gene product': semantic_similarity,
        'When treated with/exposed to/when assayed with': lambda gt, pred: category_match(gt, pred, 
            ['when treated with', 'when exposed to', 'when assayed with', 'due to']),
        'Multiple drugs And/or': lambda gt, pred: category_match(gt, pred, ['and', 'or']),
        'Cell type': semantic_similarity,
        'Comparison Allele(s) or Genotype(s)': semantic_similarity,
        'Comparison Metabolizer types': semantic_similarity
    }
    
    # Calculate scores
    results = {
        'total_samples': len(ground_truth),
        'field_scores': {},
        'overall_score': 0.0
    }
    
    # Score each field
    for field, evaluator in field_evaluators.items():
        scores = []
        for gt, pred in zip(ground_truth, predictions):
            gt_val = gt.get(field)
            pred_val = pred.get(field)
            score = evaluator(gt_val, pred_val)
            scores.append(score)
        
        results['field_scores'][field] = {
            'mean_score': sum(scores) / len(scores),
            'scores': scores
        }
    
    # Generate detailed results for each sample
    results['detailed_results'] = []
    
    for i, (gt, pred) in enumerate(zip(ground_truth, predictions)):
        sample_result = {
            'sample_id': i,
            'field_scores': {}
        }
        
        for field, evaluator in field_evaluators.items():
            gt_val = gt.get(field)
            pred_val = pred.get(field)
            score = evaluator(gt_val, pred_val)
            sample_result['field_scores'][field] = score
        
        # Add dependency validation issues
        dependency_issues = validate_all_dependencies(pred, study_parameters)
        sample_result['dependency_issues'] = dependency_issues
        
        # Apply dependency penalty to relevant field scores only
        if dependency_issues:
            penalty_per_issue = 0.05  # 5% penalty per dependency issue
            total_penalty = min(len(dependency_issues) * penalty_per_issue, 0.3)  # Cap at 30% penalty
            
            # Define which fields to penalize based on the type of issue
            fields_to_penalize = set()
            for issue in dependency_issues:
                if "Gene" in issue or "gene" in issue:
                    fields_to_penalize.update(['Gene', 'Gene/gene product'])
                elif "Variant" in issue or "variant" in issue:
                    fields_to_penalize.update(['Variant/Haplotypes', 'Comparison Allele(s) or Genotype(s)'])
                elif "Direction" in issue or "Associated" in issue:
                    fields_to_penalize.update(['Direction of effect', 'Is/Is Not associated'])
                elif "Functional" in issue:
                    fields_to_penalize.update(['Functional terms', 'Gene/gene product'])
                elif "rsID" in issue or "star allele" in issue:
                    fields_to_penalize.add('Variant/Haplotypes')
                else:
                    # For other issues, penalize all fields
                    fields_to_penalize.update(sample_result['field_scores'].keys())
            
            for field in fields_to_penalize:
                if field in sample_result['field_scores']:
                    original_score = sample_result['field_scores'][field]
                    penalized_score = original_score * (1 - total_penalty)
                    sample_result['field_scores'][field] = penalized_score
        
        results['detailed_results'].append(sample_result)
    
    # Recalculate field scores and overall score after penalties
    field_names = list(field_evaluators.keys())
    for field in field_names:
        field_scores = [sample['field_scores'][field] for sample in results['detailed_results']]
        results['field_scores'][field] = {
            'mean_score': sum(field_scores) / len(field_scores),
            'scores': field_scores
        }
    
    # Calculate overall score
    field_scores = [scores['mean_score'] for scores in results['field_scores'].values()]
    results['overall_score'] = sum(field_scores) / len(field_scores) if field_scores else 0.0
    
    return results