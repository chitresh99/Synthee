from feedback.eda_main import run_comprehensive_eda


def main():
    csv_file_path = "generated_dataset_batched.csv"

    results = run_comprehensive_eda(
        csv_file_path=csv_file_path, print_report=False, save_feedback=True
    )

    if results:
        print("\nEDA analysis completed successfully!")
        print(f"Overall Quality Score: {results['quality_scores']['overall']:.1f}%")
    else:
        print("EDA analysis failed. Please check your CSV file path and format.")


if __name__ == "__main__":
    main()
