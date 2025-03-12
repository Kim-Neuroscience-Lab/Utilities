# test/test_roi_area_analyzer.py

"""
Test script for roi area analysis on test directory.
"""

import argparse
from pathlib import Path
from src.analysis.roi_area_analyzer import ROIAreaAnalyzer


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test ROI area analysis")
    parser.add_argument(
        "--use-gpu", action="store_true", help="Use GPU acceleration if available"
    )
    args = parser.parse_args()

    # Path to test directory
    test_dir = Path(
        "/Volumes/euiseokdataUCSC_1/Matt_Jacobs/Images_and_Data/H2B_quantification/01_first_pass_test_set/p 6/m_733"
    )

    # Initialize analyzer with test directory
    analyzer = ROIAreaAnalyzer(str(test_dir), use_gpu=args.use_gpu)

    # Run analysis on all directories
    print("\nAnalyzing test directory structure...")
    results = analyzer.analyze_all_directories()

    if not results:
        print("No ROIs found to analyze!")
        return

    # Print results for each animal
    print("\nAnalysis Results:")
    print("-" * 20)
    for animal_id, df in results.items():
        print(f"\nAnimal: {animal_id}")
        print(f"Total ROIs analyzed: {len(df)}")
        print(f"Number of unique regions: {df['region_name'].nunique()}")
        print(f"Number of segments: {df['segment_id'].nunique()}")
        print("\nRegion breakdown:")
        print(df.groupby("region_name", observed=True).size())


if __name__ == "__main__":
    main()
