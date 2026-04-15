#!/usr/bin/env python3
"""
Merge individual evaluation result JSON files into the main results file.
Preserves existing results and adds/updates with new individual results.
"""
import json
import sys
from pathlib import Path

def merge_results(individual_dir: str, main_results_file: str):
    """
    Merge all individual JSON files into the main results file.

    Args:
        individual_dir: Directory containing individual result JSON files
        main_results_file: Path to the main results JSON file
    """
    individual_path = Path(individual_dir)
    main_path = Path(main_results_file)

    # Load existing main results
    main_results = {}
    if main_path.exists():
        print(f"Loading existing results from: {main_path}")
        with open(main_path, 'r') as f:
            main_results = json.load(f)
        print(f"  Found {len(main_results)} existing results")
    else:
        print(f"No existing results file at {main_path}, will create new one")

    # Find all individual JSON files
    if not individual_path.exists():
        print(f"ERROR: Directory {individual_path} does not exist")
        return 1

    json_files = sorted(individual_path.glob("*.json"))
    print(f"\nFound {len(json_files)} individual result files in {individual_path}")

    if not json_files:
        print("No files to merge!")
        return 0

    # Merge individual results into main results
    updated = 0
    new = 0
    errors = 0

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Each individual file should have format: {run_name: {results}}
            # Since eval_checkpoints_proper.py writes the whole dict
            for run_name, result in data.items():
                if run_name in main_results:
                    # Check if we're updating an error with a success
                    if "error" in main_results[run_name] and "error" not in result:
                        print(f"  ✓ Updating {run_name}: error -> success")
                        updated += 1
                    elif "error" not in main_results[run_name] and "error" in result:
                        print(f"  ⚠ Updating {run_name}: success -> error (keeping new)")
                        updated += 1
                    else:
                        updated += 1
                else:
                    new += 1

                main_results[run_name] = result

                if "error" in result:
                    errors += 1

        except Exception as e:
            print(f"  ⚠ Warning: Failed to read {json_file}: {e}")

    # Write merged results
    main_path.parent.mkdir(parents=True, exist_ok=True)
    with open(main_path, 'w') as f:
        json.dump(main_results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Merge complete!")
    print(f"  Total results: {len(main_results)}")
    print(f"  New results: {new}")
    print(f"  Updated results: {updated}")
    print(f"  Results with errors: {errors}")
    print(f"  Output file: {main_path}")
    print(f"{'='*80}")

    # Print summary statistics
    print("\nSUMMARY")
    print("=" * 81)
    print(f"{'Run':<60} {'Val Loss':>10} {'PPL':>10}")
    print("-" * 81)

    for run_name in sorted(main_results.keys()):
        result = main_results[run_name]
        if 'error' in result:
            print(f"{run_name:<60} {'ERROR':>10} {'ERROR':>10}")
        else:
            loss = result.get('validation_loss', 'N/A')
            ppl = result.get('perplexity', 'N/A')
            if isinstance(loss, (int, float)) and isinstance(ppl, (int, float)):
                print(f"{run_name:<60} {loss:>10.4f} {ppl:>10.2f}")
            else:
                print(f"{run_name:<60} {str(loss):>10} {str(ppl):>10}")

    return 0

def main():
    if len(sys.argv) != 3:
        print("Usage: python merge_eval_results.py <individual_results_dir> <main_results_file>")
        print("")
        print("Example:")
        print("  python merge_eval_results.py \\")
        print("    ../results/eval_chinchilla8_individual \\")
        print("    ../results/eval_results_chinchilla8.json")
        sys.exit(1)

    individual_dir = sys.argv[1]
    main_results_file = sys.argv[2]

    sys.exit(merge_results(individual_dir, main_results_file))

if __name__ == "__main__":
    main()
