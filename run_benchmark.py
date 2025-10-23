"""
Run benchmark with actual data
"""

from src.fa_benchmark.fa_benchmark import evaluate_functional_analysis
import json

def main():
    # Load your benchmark data
    with open('data/benchmark_annotations.json', 'r') as f:
        data = json.load(f)
    
    # Extract functional analysis annotations (var_fa_ann)
    gt_annotations = []
    pred_annotations = []
    
    for pmcid, article_data in data.items():
        if 'var_fa_ann' in article_data and article_data['var_fa_ann']:
            print(f"\n=== Processing PMCID: {pmcid} ===")
            print(f"Title: {article_data.get('title', 'N/A')}")
            print(f"Found {len(article_data['var_fa_ann'])} functional analysis annotations")
            
            for i, annotation in enumerate(article_data['var_fa_ann']):
                print(f"\n--- Annotation {i+1} ---")
                print(f"Variant: {annotation.get('Variant/Haplotypes', 'N/A')}")
                print(f"Gene: {annotation.get('Gene', 'N/A')}")
                print(f"Drug: {annotation.get('Drug(s)', 'N/A')}")
                print(f"Significance: {annotation.get('Significance', 'N/A')}")
                print(f"Phenotype Category: {annotation.get('Phenotype Category', 'N/A')}")
                
                # This is your ground truth data
                gt_annotations.append(annotation)
                
                # Create realistic LLM-generated predictions with various types of errors
                mock_pred = annotation.copy()
                
                # Simulate different types of LLM errors
                import random
                
                # 1. Sometimes get the gene wrong
                if random.random() < 0.1:  # 10% chance
                    mock_pred['Gene'] = 'CYP2D6' if mock_pred.get('Gene') != 'CYP2D6' else 'CYP3A4'
                
                # 2. Sometimes get the significance wrong
                if random.random() < 0.15:  # 15% chance
                    if mock_pred.get('Significance') == 'yes':
                        mock_pred['Significance'] = 'no'
                    elif mock_pred.get('Significance') == 'no':
                        mock_pred['Significance'] = 'yes'
                
                # 3. Sometimes get the direction of effect wrong
                if random.random() < 0.2:  # 20% chance
                    if mock_pred.get('Direction of effect') == 'increased':
                        mock_pred['Direction of effect'] = 'decreased'
                    elif mock_pred.get('Direction of effect') == 'decreased':
                        mock_pred['Direction of effect'] = 'increased'
                
                # 4. Sometimes get functional terms slightly wrong
                if random.random() < 0.25:  # 25% chance
                    if 'Functional terms' in mock_pred and mock_pred['Functional terms']:
                        if 'expression' in mock_pred['Functional terms']:
                            mock_pred['Functional terms'] = mock_pred['Functional terms'].replace('expression', 'activity')
                        elif 'activity' in mock_pred['Functional terms']:
                            mock_pred['Functional terms'] = mock_pred['Functional terms'].replace('activity', 'metabolism')
                
                # 5. Sometimes miss the variant/haplotype
                if random.random() < 0.1:  # 10% chance
                    mock_pred['Variant/Haplotypes'] = 'rs123456'  # Wrong variant
                
                # 6. Sometimes get phenotype category wrong
                if random.random() < 0.15:  # 15% chance
                    categories = ['efficacy', 'toxicity', 'dosage', 'metabolism/PK', 'pd', 'other']
                    current_cat = mock_pred.get('Phenotype Category', 'other')
                    other_cats = [cat for cat in categories if cat != current_cat]
                    if other_cats:
                        mock_pred['Phenotype Category'] = random.choice(other_cats)
                
                # 7. Sometimes miss the drug
                if random.random() < 0.1:  # 10% chance
                    mock_pred['Drug(s)'] = None
                
                # Print full comparison
                print(f"  GROUND TRUTH:")
                for key, value in annotation.items():
                    print(f"    {key}: {value}")
                
                print(f"  LLM PREDICTION:")
                for key, value in mock_pred.items():
                    print(f"    {key}: {value}")
                
                print(f"  ---")
                
                pred_annotations.append(mock_pred)
    
    print(f"Found {len(gt_annotations)} functional analysis annotations")
    
    if len(gt_annotations) == 0:
        print("No functional analysis annotations found in the data")
        return
    
    # Run the benchmark
    results = evaluate_functional_analysis(gt_annotations, pred_annotations)
    
    print(f"\n=== INDIVIDUAL ANNOTATION SCORES ===")
    for i, sample_result in enumerate(results['detailed_results']):
        print(f"\n--- Annotation {i+1} Scores ---")
        for field, score in sample_result['field_scores'].items():
            print(f"  {field:25}: {score:.3f}")
    
    print(f"\n=== OVERALL BENCHMARK RESULTS ===")
    print(f"Overall Score: {results['overall_score']:.3f}")
    print(f"Total Samples: {results['total_samples']}")
    
    print(f"\n=== AVERAGE FIELD SCORES ===")
    for field, scores in results['field_scores'].items():
        print(f"{field:25}: {scores['mean_score']:.3f}")

if __name__ == "__main__":
    main()
