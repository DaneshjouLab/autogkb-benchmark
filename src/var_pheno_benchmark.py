from typing import List, Dict, Any, Tuple, Set
from dataclasses import dataclass
import re


class PhenotypeAnnotationBenchmark:
    """Benchmark for evaluating phenotype annotation predictions."""

    # Fields to compare (excluding metadata fields)
    CORE_FIELDS = [
        "Variant/Haplotypes",
        "Gene",
        "Drug(s)",
        "Phenotype Category",
        "Alleles",
        "Is/Is Not associated",
        "Direction of effect",
        "Phenotype",
        "When treated with/exposed to/when assayed with",
        "Comparison Allele(s) or Genotype(s)",
    ]

    # Fields with weighted importance
    FIELD_WEIGHTS = {
        "Phenotype": 2.0,
        "Drug(s)": 1.5,
        "Direction of effect": 2.0,
        "Alleles": 1.5,
        "Is/Is Not associated": 1.0,
        "Variant/Haplotypes": 1.0,
        "Gene": 1.0,    
        "Phenotype Category": 0.5,
        "When treated with/exposed to/when assayed with": 0.5,
        "Comparison Allele(s) or Genotype(s)": 1.0,
    }

    def __init__(self, matching_threshold: float = 0.7):
        """
        Initialize benchmark.

        Args:
            matching_threshold: Minimum score to consider a match (0-1)
        """
        self.matching_threshold = matching_threshold

    def _normalize_value(self, value: Any) -> str:
        """Normalize a field value for comparison."""
        if value is None:
            return ""

        # Convert to string and normalize
        s = str(value).lower().strip()

        # Remove extra whitespace
        s = re.sub(r'\s+', ' ', s)

        # Remove punctuation variations
        s = re.sub(r'[,;]+', '', s)

        return s

    def _compare_field(self, pred_value: Any, gt_value: Any) -> float:
        """
        Compare two field values and return similarity score (0-1).

        Args:
            pred_value: Predicted value
            gt_value: Ground truth value

        Returns:
            Similarity score between 0 and 1
        """
        pred_norm = self._normalize_value(pred_value)
        ground_truth_norm = self._normalize_value(gt_value)

        # Both empty or null
        if not pred_norm and not ground_truth_norm:
            return 1.0

        # One empty, one not
        if not pred_norm or not ground_truth_norm:
            return 0.0

        # Exact match
        if pred_norm == ground_truth_norm:
            return 1.0

        # Check if one contains the other (useful for partial matches)
        if pred_norm in ground_truth_norm or ground_truth_norm in pred_norm:
            return 0.8

        #The Jaccard index is particularly useful when the presence or absence of elements 
        # in the sets is more important than their frequency or order.
        # could be used to help check for multiple entries put in one annotation?
        pred_tokens = set(pred_norm.split())
        gt_tokens = set(ground_truth_norm.split())

        if pred_tokens and gt_tokens:
            intersection = len(pred_tokens & gt_tokens)
            union = len(pred_tokens | gt_tokens)
            jaccard = intersection / union if union > 0 else 0.0
            return jaccard

        return 0.0

    def _compare_annotations(self, pred: Dict[str, Any], gt: Dict[str, Any]) -> float:
        """
        Compare a predicted annotation with a ground truth annotation.

        Args:
            pred: Predicted annotation
            gt: Ground truth annotation

        Returns:
            Float ranging from 0 - 1 denoting similarity
        """
        field_scores = {}
        weighted_sum = 0.0
        total_weight = 0.0

        for field in self.CORE_FIELDS:
            weight = self.FIELD_WEIGHTS.get(field, 1.0)
            similarity = self._compare_field(pred.get(field), gt.get(field))

            field_scores[field] = similarity
            weighted_sum += similarity * weight
            total_weight += weight

        # Calculate weighted average
        matching_score = weighted_sum / total_weight

        return matching_score

    def _find_best_matches(
        self,
        predictions: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]]
    ) -> List[Tuple[int, int, float]]:
        """
        Find best matches between predictions and ground truths.

        Returns:
            List of (pred_idx, gt_idx, score) tuples sorted by score descending
        """
        matches = []

        for pred_idx, pred in enumerate(predictions):
            for gt_idx, gt in enumerate(ground_truths):
                match_score = self._compare_annotations(pred, gt)
                if match_score >= self.matching_threshold:
                    matches.append((pred_idx, gt_idx, match_score))

        # Sort by score descending
        matches.sort(key=lambda x: x[2], reverse=True)

        return matches

    def evaluate(
        self,
        predictions: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]]
    ) -> float:
        """
        Evaluate predictions against ground truths and return similarity score.

        Handles many-to-many matching where multiple predicted annotations
        might correspond to a single ground truth or vice versa.

        Args:
            predictions: List of predicted annotations
            ground_truths: List of ground truth annotations

        Returns:
            Similarity score between 0 and 1
        """
        if not ground_truths or not predictions:
            return 0.0

        # Find all potential matches
        all_matches = self._find_best_matches(predictions, ground_truths)

        # Greedily assign matches (allowing many-to-one mapping)
        matched_preds: Set[int] = set()
        matched_gts: Set[int] = set()
        match_scores = []

        for pred_idx, gt_idx, score in all_matches:
            # Allow multiple predictions to match same ground truth (many-to-one)
            # but each prediction can only match once (one-to-one from pred side)
            if pred_idx not in matched_preds:
                matched_preds.add(pred_idx)
                matched_gts.add(gt_idx)
                match_scores.append(score)

        # Calculate average similarity across all ground truths
        # Matched GTs contribute their match score
        # Unmatched GTs contribute 0
        total_score = sum(match_scores)
        total_possible = len(ground_truths)

        return total_score / total_possible


def benchmark_phenotype_annotations(
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    matching_threshold: float = 0.7
) -> float:
    """
    Benchmark phenotype annotations and return an aggregate similarity score.

    Args:
        predictions: List of predicted annotation dictionaries
        ground_truths: List of ground truth annotation dictionaries
        matching_threshold: Minimum similarity score to consider a match (0-1)

    Returns:
        Similarity score between 0-100 representing how well predictions
        match ground truths across all fields.

    Example:
        >>> predictions = [{"Phenotype": "sensitivity", "Drug(s)": "etoposide", ...}]
        >>> ground_truths = [{"Phenotype": "sensitivity", "Drug(s)": "etoposide", ...}]
        >>> score = benchmark_phenotype_annotations(predictions, ground_truths)
        >>> print(f"Model Score: {score:.1f}/100")
    """
    benchmark = PhenotypeAnnotationBenchmark(matching_threshold=matching_threshold)
    similarity = benchmark.evaluate(predictions, ground_truths)

    # Return as 0-100 scale
    return similarity * 100


if __name__ == "__main__":
    model_prediction = [
        {
        "Variant Annotation ID": 748448812,
        "Variant/Haplotypes": "rs6539870",
        "Gene": None,
        "Drug(s)": "etoposide",
        "PMID": 17537913,
        "Phenotype Category": "Other",
        "Significance": "yes",
        "Notes": None,
        "Sentence": "Genotype GG is associated with increased sensitivity to etoposide when exposed to etoposide as compared to genotypes AA + AG.",
        "Alleles": "GG",
        "Specialty Population": None,
        "Metabolizer types": None,
        "isPlural": "Is",
        "Is/Is Not associated": "Associated with",
        "Direction of effect": "increased",
        "Side effect/efficacy/other": None,
        "Phenotype": "Other:sensitivity to etoposide",
        "Multiple phenotypes And/or": None,
        "When treated with/exposed to/when assayed with": "when exposed to",
        "Multiple drugs And/or": None,
        "Population types": None,
        "Population Phenotypes or diseases": None,
        "Multiple phenotypes or diseases And/or": None,
        "Comparison Allele(s) or Genotype(s)": "AA + AG",
        "Comparison Metabolizer types": None,
        "PMID_norm": "17537913",
        "Variant Annotation ID_norm": "748448812"
      },
         {
        "Variant Annotation ID": 1453076180,
        "Variant/Haplotypes": "NUDT15*3",
        "Gene": "NUDT15",
        "Drug(s)": "mercaptopurine",
        "PMID": 40099566,
        "Phenotype Category": "Toxicity",
        "Significance": "no",
        "Notes": "\"case of thiopurine toxicity resulting in severe myelosuppression, hepatotoxicity and alopecia in an individual with homozygous *3/*3 loss-of-function alleles in the NUDT15 gene\"",
        "Sentence": "NUDT15 *3/*3 is associated with increased severity of Myelosuppression, Toxic liver disease and Alopecia when treated with mercaptopurine in women with Leukemia, Promyelocytic, Acute.",
        "Alleles": "*3/*3",
        "Specialty Population": None,
        "Metabolizer types": None,
        "isPlural": "Is",
        "Is/Is Not associated": "Associated with",
        "Direction of effect": "increased",
        "Side effect/efficacy/other": "severity of",
        "Phenotype": "Side Effect:Myelosuppression, Side Effect:Toxic liver disease, Side Effect:Alopecia",
        "Multiple phenotypes And/or": "and",
        "When treated with/exposed to/when assayed with": "when treated with",
        "Multiple drugs And/or": None,
        "Population types": "in women with",
        "Population Phenotypes or diseases": "Other:Leukemia, Promyelocytic, Acute",
        "Multiple phenotypes or diseases And/or": None,
        "Comparison Allele(s) or Genotype(s)": None,
        "Comparison Metabolizer types": None,
        "PMID_norm": "40099566",
        "Variant Annotation ID_norm": "1453076180"
      }
    ]
    ground_truth = [
        {
        "Variant Annotation ID": 748448812,
        "Variant/Haplotypes": "rs6539870",
        "Gene": None,
        "Drug(s)": "etoposide",
        "PMID": 17537913,
        "Phenotype Category": "Other",
        "Significance": "yes",
        "Notes": None,
        "Sentence": "Genotype GG is associated with increased sensitivity to etoposide when exposed to etoposide as compared to genotypes AA + AG.",
        "Alleles": "GG",
        "Specialty Population": None,
        "Metabolizer types": None,
        "isPlural": "Is",
        "Is/Is Not associated": "Associated with",
        "Direction of effect": "increased",
        "Side effect/efficacy/other": None,
        "Phenotype": "Other:sensitivity to etoposide",
        "Multiple phenotypes And/or": None,
        "When treated with/exposed to/when assayed with": "when exposed to",
        "Multiple drugs And/or": None,
        "Population types": None,
        "Population Phenotypes or diseases": None,
        "Multiple phenotypes or diseases And/or": None,
        "Comparison Allele(s) or Genotype(s)": "AA + AG",
        "Comparison Metabolizer types": None,
        "PMID_norm": "17537913",
        "Variant Annotation ID_norm": "748448812"
      },
         {
        "Variant Annotation ID": 1453076180,
        "Variant/Haplotypes": "NUDT15*3",
        "Gene": "NUDT15",
        "Drug(s)": "mercaptopurine",
        "PMID": 40099566,
        "Phenotype Category": "Toxicity",
        "Significance": "no",
        "Notes": "\"case of thiopurine toxicity resulting in severe myelosuppression, hepatotoxicity and alopecia in an individual with homozygous *3/*3 loss-of-function alleles in the NUDT15 gene\"",
        "Sentence": "NUDT15 *3/*3 is associated with increased severity of Myelosuppression, Toxic liver disease and Alopecia when treated with mercaptopurine in women with Leukemia, Promyelocytic, Acute.",
        "Alleles": "*3/*3",
        "Specialty Population": None,
        "Metabolizer types": None,
        "isPlural": "Is",
        "Is/Is Not associated": "Associated with",
        "Direction of effect": "increased",
        "Side effect/efficacy/other": "severity of",
        "Phenotype": "Side Effect:Myelosuppression, Side Effect:Toxic liver disease, Side Effect:Alopecia",
        "Multiple phenotypes And/or": "and",
        "When treated with/exposed to/when assayed with": "when treated with",
        "Multiple drugs And/or": None,
        "Population types": "in women with",
        "Population Phenotypes or diseases": "Other:Leukemia, Promyelocytic, Acute",
        "Multiple phenotypes or diseases And/or": None,
        "Comparison Allele(s) or Genotype(s)": None,
        "Comparison Metabolizer types": None,
        "PMID_norm": "40099566",
        "Variant Annotation ID_norm": "1453076180"
      }
    ]
    
    print(benchmark_phenotype_annotations(model_prediction, ground_truth))