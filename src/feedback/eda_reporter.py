import pandas as pd


class EDAReporter:
    def __init__(self, analysis_results):
        self.results = analysis_results
    
    def print_basic_info(self):
        info = self.results['basic_info']
        print("="*80)
        print("COMPREHENSIVE EDA ANALYSIS FOR SYNTHETIC DATASET")
        print("="*80)
        print(f"\nDataset Shape: {info['shape'][0]} rows × {info['shape'][1]} columns")
        print(f"Memory Usage: {info['memory_usage_kb']:.2f} KB")
        print(f"File Size: {info['file_size_kb']:.2f} KB")
    
    def print_column_analysis(self):
        print("\nCOLUMN ANALYSIS")
        print("-" * 40)
        for i, (col, info) in enumerate(self.results['column_info'].items(), 1):
            print(f"{i:2d}. {col:<25} | {info['dtype']:<15} | Non-null: {info['non_null_count']}")
    
    def print_quality_assessment(self):
        print("\nDATA QUALITY ASSESSMENT")
        print("-" * 40)
        
        qa = self.results['quality_assessment']
        
        missing_sum = sum(qa['missing_values'].values())
        if missing_sum > 0:
            print("Missing Values:")
            for col, count in qa['missing_values'].items():
                if count > 0:
                    print(f"  {col}: {count} ({qa['missing_percent'][col]:.2f}%)")
        else:
            print("No missing values found")
        
        print(f"Duplicate Rows: {qa['duplicate_count']} ({qa['duplicate_percent']:.2f}%)")
        
        if qa['empty_strings']:
            print("Empty String Values:")
            for col, count in qa['empty_strings'].items():
                print(f"  {col}: {count}")
        else:
            print("No empty string values found")
    
    def print_statistical_summary(self):
        print("\nSTATISTICAL SUMMARY")
        print("-" * 40)
        
        summary = self.results['statistical_summary']
        
        if 'numeric' in summary:
            print("Numerical Columns Summary:")
            df_desc = pd.DataFrame(summary['numeric']).round(2)
            print(df_desc)
            
            print("\nOutlier Analysis:")
            for col, outlier_info in summary['outliers'].items():
                print(f"  {col}: {outlier_info['count']} outliers ({outlier_info['percent']:.1f}%)")
        
        if 'categorical' in summary:
            print("\nCategorical Columns Summary:")
            for col, info in summary['categorical'].items():
                most_common = info['most_common'] or 'N/A'
                print(f"  {col}: {info['unique_count']} unique | Most common: '{most_common}'")
    
    def print_uniqueness_analysis(self):
        print("\nUNIQUENESS ANALYSIS")
        print("-" * 40)
        for col, info in self.results['column_info'].items():
            unique_percent = info['unique_percent']
            if unique_percent > 95:
                status = "Potential ID"
            elif unique_percent > 10:
                status = "Normal"
            else:
                status = "Low variety"
            print(f"{col:<25}: {info['unique_count']:4d} unique ({unique_percent:5.1f}%) {status}")
    
    def print_value_distributions(self):
        print("\nVALUE DISTRIBUTION ANALYSIS")
        print("-" * 40)
        
        if 'categorical' in self.results['statistical_summary']:
            for col, info in self.results['statistical_summary']['categorical'].items():
                if info['unique_count'] <= 20:
                    print(f"\n{col} - Value Distribution:")
                    total_rows = self.results['basic_info']['shape'][0]
                    for value, count in info['value_counts'].items():
                        percentage = (count / total_rows) * 100
                        bar = "█" * min(int(percentage / 2), 50)
                        print(f"  {str(value):<20} | {count:4d} ({percentage:5.1f}%) {bar}")
    
    def print_consistency_checks(self):
        print("\nDATA CONSISTENCY CHECKS")
        print("-" * 40)
        
        issues = self.results['consistency_issues']
        if issues:
            print("Consistency Issues Found:")
            for issue in issues:
                print(f"  {issue}")
        else:
            print("No major consistency issues detected")
    
    def print_correlation_analysis(self):
        print("\nCOLUMN RELATIONSHIP ANALYSIS")
        print("-" * 40)
        
        correlations = self.results['correlations']
        if correlations:
            print("High Correlations (|r| > 0.7):")
            for corr in correlations:
                print(f"  {corr['col1']} ↔ {corr['col2']}: {corr['correlation']:.3f}")
        else:
            print("No high correlations found between numerical columns")
    
    def print_range_analysis(self):
        print("\nREALISTIC VALUE RANGES CHECK")
        print("-" * 40)
        
        issues = self.results['range_issues']
        if issues:
            print("Potential Range Issues:")
            for issue in issues:
                print(f"  {issue}")
        else:
            print("Value ranges appear realistic")
    
    def print_quality_scores(self):
        print("\nDATASET QUALITY SCORE")
        print("-" * 40)
        
        scores = self.results['quality_scores']
        print(f"Completeness Score:   {scores['completeness']:.1f}%")
        print(f"Uniqueness Score:     {scores['uniqueness']:.1f}%")
        print(f"Consistency Score:    {scores['consistency']:.1f}%")
        print(f"Range Validity Score: {scores['range_validity']:.1f}%")
        print(f"Overall Quality Score: {scores['overall']:.1f}%")
    
    def print_recommendations(self):
        print("\nRECOMMENDATIONS FOR IMPROVEMENT")
        print("-" * 40)
        
        recommendations = self.results['recommendations']
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec}")
        else:
            print("Dataset quality looks good! No major improvements needed.")
    
    def print_full_report(self):
        self.print_basic_info()
        self.print_column_analysis()
        self.print_quality_assessment()
        self.print_statistical_summary()
        self.print_uniqueness_analysis()
        self.print_value_distributions()
        self.print_consistency_checks()
        self.print_correlation_analysis()
        self.print_range_analysis()
        self.print_quality_scores()
        self.print_recommendations()
        
        print("\n" + "="*80)
        print("EDA ANALYSIS COMPLETE")
        print("="*80)