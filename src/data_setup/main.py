"""
Main data setup module for AutoGKB Benchmark
"""

import json
from pathlib import Path
from .download import download_data


def setup_data():
    """Main data setup function"""
    
    print("Setting up benchmark data...")
    
    # Download data if needed
    download_data()
    
    # Create sample data if it doesn't exist
    create_sample_data()
    
    print("Data setup complete!")


def create_sample_data():
    """Create sample data files for testing"""
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Sample ground truth data
    sample_gt = [
        {
            "Variant/Haplotypes": "rs2784917",
            "Gene": "WNT5B",
            "Drug(s)": None,
            "PMID": "17537913",
            "Phenotype Category": "Other",
            "Significance": "yes",
            "Alleles": "AA",
            "Direction of effect": "increased",
            "Functional terms": "expression of",
            "Gene/gene product": "WNT5B",
            "Comparison Allele(s) or Genotype(s)": "AG + GG"
        }
    ]
    
    # Sample predictions data
    sample_pred = [
        {
            "Variant/Haplotypes": "rs2784917",
            "Gene": "WNT5B",
            "Drug(s)": None,
            "PMID": "17537913",
            "Phenotype Category": "Other",
            "Significance": "yes",
            "Alleles": "AA",
            "Direction of effect": "increased",
            "Functional terms": "expression",  # Slightly different
            "Gene/gene product": "WNT5B",
            "Comparison Allele(s) or Genotype(s)": "AG + GG"
        }
    ]
    
    # Save sample data
    gt_path = data_dir / "functional_analysis_gt.json"
    pred_path = data_dir / "functional_analysis_pred.json"
    
    if not gt_path.exists():
        with open(gt_path, 'w') as f:
            json.dump(sample_gt, f, indent=2)
        print(f"Created sample ground truth: {gt_path}")
    
    if not pred_path.exists():
        with open(pred_path, 'w') as f:
            json.dump(sample_pred, f, indent=2)
        print(f"Created sample predictions: {pred_path}")


if __name__ == "__main__":
    setup_data()