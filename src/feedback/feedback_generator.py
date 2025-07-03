class FeedbackGenerator:
    def __init__(self, analysis_results):
        self.results = analysis_results
    
    def generate_feedback_prompt(self):
        basic_info = self.results['basic_info']
        quality_scores = self.results['quality_scores']
        quality_assessment = self.results['quality_assessment']
        
        prompt = f"""
        SYNTHETIC DATASET QUALITY ANALYSIS REPORT

        Dataset Overview:
        - Shape: {basic_info['shape'][0]} rows × {basic_info['shape'][1]} columns
        - Overall Quality Score: {quality_scores['overall']:.1f}%
        - Completeness: {quality_scores['completeness']:.1f}%
        - Missing Values: {sum(quality_assessment['missing_values'].values())}
        - Duplicate Rows: {quality_assessment['duplicate_count']}
        - Consistency Issues: {len(self.results['consistency_issues'])}
        - Range Issues: {len(self.results['range_issues'])}

        Column Information:
        {self._format_column_info()}

        Key Issues Found:
        {self._format_recommendations()}
        """

        return prompt
    
    def _format_column_info(self):
        return "\n".join([f"{col}: {info['dtype']}" for col, info in self.results['column_info'].items()])
    
    def _format_recommendations(self):
        recommendations = self.results['recommendations']
        return "\n".join([f"- {rec}" for rec in recommendations]) if recommendations else "No major issues detected"
    
    def save_feedback_prompt(self, filename='eda_feedback_prompt.txt'):
        prompt = self.generate_feedback_prompt()
        with open(filename, 'w') as f:
            f.write(prompt)
        return filename