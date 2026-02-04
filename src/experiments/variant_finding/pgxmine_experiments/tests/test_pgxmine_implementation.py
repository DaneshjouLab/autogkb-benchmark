#!/usr/bin/env python3
"""
Quick test of PGxMine implementation on a single article.
"""

from src.modules.variant_finding.variant_extractor import VariantExtractor

# Test article PMC5508045: has 4 rsID variants
# Expected: ["rs9923231", "rs887829", "rs2108622", "rs1057910"]
test_pmcid = "PMC5508045"

print(f"\n{'='*60}")
print(f"Testing PGxMine implementations on {test_pmcid}")
print(f"Expected variants: rs9923231, rs887829, rs2108622, rs1057910")
print(f"{'='*60}\n")

methods = [
    "pgxmine_context_aware",
    "pgxmine_normalized",
    "pgxmine_full"
]

for method in methods:
    print(f"\n{method}:")
    print("-" * 60)
    try:
        extractor = VariantExtractor(method)
        variants = extractor.get_variants(test_pmcid)
        print(f"✓ Extracted {len(variants)} variants:")
        for v in sorted(variants):
            print(f"  - {v}")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*60}")
print("Test complete!")
print(f"{'='*60}\n")
