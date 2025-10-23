from typing import Dict, List, Any
from difflib import SequenceMatcher


def evaluate_functional_analysis(ground_truth: List[Dict[str, Any]], 
                                predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
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
    
    # Field evaluation functions
    def exact_match(gt_val: Any, pred_val: Any) -> float:
        """1:1 exact match"""
        if gt_val is None and pred_val is None:
            return 1.0
        if gt_val is None or pred_val is None:
            return 0.0
        return 1.0 if str(gt_val).strip().lower() == str(pred_val).strip().lower() else 0.0
    
    def semantic_similarity(gt_val: Any, pred_val: Any) -> float:
        """Semantic similarity using sequence matching"""
        if gt_val is None and pred_val is None:
            return 1.0
        if gt_val is None or pred_val is None:
            return 0.0
        
        gt_str = str(gt_val).strip().lower()
        pred_str = str(pred_val).strip().lower()
        
        if gt_str == pred_str:
            return 1.0
        
        return SequenceMatcher(None, gt_str, pred_str).ratio()
    
    def variant_coverage(gt_variants: str, pred_variants: str) -> float:
        """Evaluate variant coverage - penalize missing, don't penalize extra"""
        if not gt_variants or not pred_variants:
            return 1.0 if not gt_variants and not pred_variants else 0.0
        
        # Simple coverage check
        gt_lower = gt_variants.lower()
        pred_lower = pred_variants.lower()
        
        if gt_lower in pred_lower or pred_lower in gt_lower:
            return 1.0
        
        return SequenceMatcher(None, gt_lower, pred_lower).ratio()
    
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
    
    # Calculate overall score
    field_scores = [scores['mean_score'] for scores in results['field_scores'].values()]
    results['overall_score'] = sum(field_scores) / len(field_scores) if field_scores else 0.0
    
    return results