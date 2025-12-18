"""
Example usage of the benchmark system.
This script demonstrates how to run benchmarks and analyze results.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from .run_benchmark import (
    run_single_benchmark,
    run_all_benchmarks,
    display_mismatches,
    display_all_mismatches
)


def example_single_file():
    """Example: Run benchmark on a single file."""
    print("=" * 80)
    print("EXAMPLE 1: Single File Benchmark")
    print("=" * 80)

    gt_file = Path('data/benchmark_annotations/PMC5508045.json')
    prop_file = Path('data/proposed_annotations/PMC5508045.json')

    if not gt_file.exists() or not prop_file.exists():
        print("Error: Example files not found. Make sure data directories exist.")
        return

    result = run_single_benchmark(gt_file, prop_file, verbose=True)

    print(f"\nFinal Score: {result['overall_score']:.3f}")
    print(f"PMID: {result['pmid']}")
    print(f"Title: {result['title']}")

    # Show benchmark-specific scores
    print("\nBenchmark Scores:")
    for name, benchmark in result['benchmarks'].items():
        if 'overall_score' in benchmark:
            print(f"  {name}: {benchmark['overall_score']:.3f}")

    # Save mismatches to JSON
    print("\n" + "=" * 80)
    print("SAVING MISMATCH ANALYSIS")
    print("=" * 80)
    output_file = display_mismatches(
        result,
        output_dir=Path('data/analysis'),
        show_unmatched=True,
        show_low_scores=True
    )
    if output_file:
        print(f"Analysis saved to: {output_file}")

        # Show a snippet of the saved analysis
        import json
        with open(output_file) as f:
            analysis = json.load(f)
        print(f"\nAnalysis contains:")
        print(f"  - Overall score: {analysis['overall_score']:.3f}")
        print(f"  - Benchmarks: {len(analysis['benchmarks'])}")
        for name, bench in analysis['benchmarks'].items():
            unmatched_gt = bench.get('unmatched_ground_truth', {}).get('count', 0)
            unmatched_pred = bench.get('unmatched_predictions', {}).get('count', 0)
            print(f"    {name}: {unmatched_gt} missing, {unmatched_pred} hallucinated")


def example_all_files():
    """Example: Run benchmark on all files."""
    print("\n\n" + "=" * 80)
    print("EXAMPLE 2: All Files Benchmark")
    print("=" * 80)

    gt_dir = Path('data/benchmark_annotations')
    prop_dir = Path('data/proposed_annotations')
    output_file = Path('example_benchmark_results.json')

    if not gt_dir.exists() or not prop_dir.exists():
        print("Error: Data directories not found.")
        return

    results = run_all_benchmarks(
        gt_dir,
        prop_dir,
        output_file=output_file,
        verbose=True
    )

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total files: {results['total_files']}")
    print(f"Mean score: {results['overall_mean_score']:.3f}")
    print(f"Min score: {results['overall_min_score']:.3f}")
    print(f"Max score: {results['overall_max_score']:.3f}")

    # Show per-benchmark statistics
    print("\nPer-Benchmark Statistics:")
    for name, stats in results['per_benchmark_stats'].items():
        print(f"\n  {name}:")
        print(f"    Mean: {stats['mean_score']:.3f}")
        print(f"    Range: [{stats['min_score']:.3f}, {stats['max_score']:.3f}]")
        print(f"    Files: {stats['num_files']}")

    # Save mismatches for low-scoring files to JSON
    print("\n" + "=" * 80)
    print("SAVING ANALYSIS FOR LOW-SCORING FILES (< 0.5)")
    print("=" * 80)
    saved_files = display_all_mismatches(
        results,
        output_dir=Path('data/analysis'),
        show_only_low_scores=True,
        overall_score_threshold=0.5,
        score_threshold=0.7,
        verbose=True
    )
    print(f"\nSaved {len(saved_files)} analysis files")


def example_read_analysis_jsons():
    """Example: Read and analyze the saved JSON files."""
    print("\n\n" + "=" * 80)
    print("EXAMPLE 3: Reading Analysis JSON Files")
    print("=" * 80)

    analysis_dir = Path('data/analysis')
    if not analysis_dir.exists():
        print("No analysis files found. Run example 1 or 2 first.")
        return

    # Read all analysis files
    analysis_files = list(analysis_dir.glob('*.json'))
    print(f"Found {len(analysis_files)} analysis files")

    if not analysis_files:
        print("No analysis files found. Run example 1 or 2 first.")
        return

    import json

    # Analyze the first file
    with open(analysis_files[0]) as f:
        analysis = json.load(f)

    print(f"\nAnalyzing: {analysis['pmcid']}")
    print(f"Title: {analysis['title']}")
    print(f"Overall Score: {analysis['overall_score']:.3f}")

    # Show benchmark details
    for benchmark_name, benchmark_data in analysis['benchmarks'].items():
        print(f"\n{benchmark_name}:")
        print(f"  Score: {benchmark_data['score']:.3f}")

        if 'unmatched_ground_truth' in benchmark_data:
            unmatched_gt_count = benchmark_data['unmatched_ground_truth']['count']
            print(f"  Missing annotations: {unmatched_gt_count}")
            if unmatched_gt_count > 0:
                print("  Examples:")
                for item in benchmark_data['unmatched_ground_truth']['items'][:2]:
                    print(f"    - {item['variant']} ({item['gene']})")

        if 'unmatched_predictions' in benchmark_data:
            unmatched_pred_count = benchmark_data['unmatched_predictions']['count']
            print(f"  Hallucinated annotations: {unmatched_pred_count}")

        if 'low_scoring_fields' in benchmark_data:
            low_count = benchmark_data['low_scoring_fields']['count']
            print(f"  Low-scoring fields: {low_count}")
            if low_count > 0:
                print("  Examples:")
                for field, data in list(benchmark_data['low_scoring_fields']['fields'].items())[:3]:
                    print(f"    - {field}: {data['mean_score']:.3f}")


def example_custom_analysis():
    """Example: Custom analysis of results."""
    print("\n\n" + "=" * 80)
    print("EXAMPLE 4: Custom Analysis")
    print("=" * 80)

    gt_dir = Path('data/benchmark_annotations')
    prop_dir = Path('data/proposed_annotations')

    if not gt_dir.exists() or not prop_dir.exists():
        print("Error: Data directories not found.")
        return

    # Run benchmarks without verbose output
    results = run_all_benchmarks(gt_dir, prop_dir, verbose=False)

    # Analyze results
    print(f"Processed {results['total_files']} files")

    # Find files with perfect scores
    perfect_files = [
        r for r in results['individual_results']
        if r['overall_score'] >= 0.95
    ]
    print(f"\nFiles with score >= 0.95: {len(perfect_files)}")
    for r in perfect_files:
        print(f"  - {r['pmcid']}: {r['overall_score']:.3f}")

    # Find files with poor scores
    poor_files = [
        r for r in results['individual_results']
        if r['overall_score'] < 0.3
    ]
    print(f"\nFiles with score < 0.3: {len(poor_files)}")
    for r in poor_files:
        print(f"  - {r['pmcid']}: {r['overall_score']:.3f}")

    # Analyze which benchmarks are most problematic
    print("\n" + "=" * 80)
    print("Most Problematic Benchmarks:")
    print("=" * 80)
    for name, stats in results['per_benchmark_stats'].items():
        print(f"{name}: Mean = {stats['mean_score']:.3f}")
        if stats['mean_score'] < 0.5:
            print(f"  ⚠️ This benchmark needs attention!")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run benchmark examples')
    parser.add_argument(
        '--example',
        type=int,
        choices=[1, 2, 3, 4],
        help='Which example to run (1=single file, 2=all files, 3=read analysis JSONs, 4=custom analysis)'
    )
    args = parser.parse_args()

    if args.example == 1:
        example_single_file()
    elif args.example == 2:
        example_all_files()
    elif args.example == 3:
        example_read_analysis_jsons()
    elif args.example == 4:
        example_custom_analysis()
    else:
        # Run first example by default
        example_single_file()
        # Uncomment to run all examples (can be slow)
        # example_all_files()
        # example_read_analysis_jsons()
        # example_custom_analysis()
