from .data_quality_analyzer import DataQualityAnalyzer
from .eda_reporter import EDAReporter
from .feedback_generator import FeedbackGenerator


def run_comprehensive_eda(csv_file_path, print_report=True, save_feedback=True):
    try:
        analyzer = DataQualityAnalyzer(csv_file_path)
        results = analyzer.run_full_analysis()

        if print_report:
            reporter = EDAReporter(results)
            reporter.print_full_report()

        if save_feedback:
            feedback_gen = FeedbackGenerator(results)
            filename = feedback_gen.save_feedback_prompt()
            print(f"\nFeedback prompt saved to '{filename}'")

        return results

    except Exception as e:
        print(f"Error during EDA analysis: {e}")
        return None
