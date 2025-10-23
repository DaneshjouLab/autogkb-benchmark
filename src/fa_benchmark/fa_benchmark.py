from typing import Dict, List, Any
from difflib import SequenceMatcher
import numpy as np
from sentence_transformers import SentenceTransformer


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
    
    # PubMedBERT model init for semantic similarity
    print("Loading PubMedBERT model...")
    model = SentenceTransformer('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
    
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
    
    # Calculate overall score
    field_scores = [scores['mean_score'] for scores in results['field_scores'].values()]
    results['overall_score'] = sum(field_scores) / len(field_scores) if field_scores else 0.0
    
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
        
        results['detailed_results'].append(sample_result)
    
    return results