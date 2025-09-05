import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")


class DataQualityAnalyzer:
    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path
        self.df = self._load_csv_robust()
        self.results = {}

    def _load_csv_robust(self):
        try:
            df = pd.read_csv(self.csv_file_path)
            print(
                f"CSV loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns"
            )
            return df
        except pd.errors.ParserError as e:
            print(f"CSV parsing error: {e}")
            print("Attempting robust parsing methods...")

            try:
                df = pd.read_csv(
                    self.csv_file_path, error_bad_lines=False, warn_bad_lines=True
                )
                print(
                    f"CSV loaded with bad lines skipped: {df.shape[0]} rows and {df.shape[1]} columns"
                )
                return df
            except:
                pass

            try:
                df = pd.read_csv(self.csv_file_path, sep=None, engine="python")
                print(
                    f"CSV loaded with auto-detected separator: {df.shape[0]} rows and {df.shape[1]} columns"
                )
                return df
            except:
                pass

            try:
                df = pd.read_csv(self.csv_file_path, quoting=3)  # QUOTE_NONE
                print(
                    f"CSV loaded with no quoting: {df.shape[0]} rows and {df.shape[1]} columns"
                )
                return df
            except:
                pass

            try:
                df = self._fix_csv_and_load()
                print(
                    f"CSV loaded after manual fixing: {df.shape[0]} rows and {df.shape[1]} columns"
                )
                return df
            except Exception as fix_error:
                print(f"All parsing methods failed: {fix_error}")
                raise

    def _fix_csv_and_load(self):
        import csv
        import io

        print("Analyzing CSV structure...")

        with open(self.csv_file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        if len(lines) < 2:
            raise ValueError("CSV file has insufficient data")

        header_line = lines[0].strip()
        expected_columns = len(header_line.split(","))
        print(f"Expected columns from header: {expected_columns}")

        problematic_lines = []
        for i, line in enumerate(lines[1:], 2):
            field_count = len(line.split(","))
            if field_count != expected_columns:
                problematic_lines.append((i, field_count))
                if len(problematic_lines) <= 5:
                    print(
                        f"Line {i}: has {field_count} fields (expected {expected_columns})"
                    )

        if problematic_lines:
            print(f"Found {len(problematic_lines)} problematic lines")

            fixed_lines = [lines[0]]

            for line in lines[1:]:
                fields = line.strip().split(",")
                if len(fields) > expected_columns:
                    fields = fields[:expected_columns]
                elif len(fields) < expected_columns:
                    fields.extend([""] * (expected_columns - len(fields)))

                fixed_lines.append(",".join(fields) + "\n")

            fixed_csv = "".join(fixed_lines)
            df = pd.read_csv(io.StringIO(fixed_csv))
            return df

        raise ValueError("Could not identify CSV structure issues")

    def get_basic_info(self):
        rows, cols = self.df.shape
        return {
            "shape": (rows, cols),
            "memory_usage_kb": self.df.memory_usage(deep=True).sum() / 1024,
            "file_size_kb": len(self.df.to_csv()) / 1024,
        }

    def analyze_columns(self):
        column_info = {}
        for col in self.df.columns:
            column_info[col] = {
                "dtype": str(self.df[col].dtype),
                "non_null_count": self.df[col].count(),
                "unique_count": self.df[col].nunique(),
                "unique_percent": (self.df[col].nunique() / len(self.df)) * 100,
            }
        return column_info

    def assess_data_quality(self):
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100

        duplicate_count = self.df.duplicated().sum()

        empty_strings = {}
        for col in self.df.select_dtypes(include=["object"]).columns:
            if self.df[col].dtype == "object":
                empty_count = (self.df[col].str.strip() == "").sum()
                if empty_count > 0:
                    empty_strings[col] = empty_count

        return {
            "missing_values": missing_data.to_dict(),
            "missing_percent": missing_percent.to_dict(),
            "duplicate_count": duplicate_count,
            "duplicate_percent": (duplicate_count / len(self.df)) * 100,
            "empty_strings": empty_strings,
        }

    def get_statistical_summary(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df.select_dtypes(include=["object"]).columns

        summary = {}

        if len(numeric_cols) > 0:
            summary["numeric"] = self.df[numeric_cols].describe().to_dict()
            summary["outliers"] = self._detect_outliers(numeric_cols)

        if len(categorical_cols) > 0:
            summary["categorical"] = {}
            for col in categorical_cols:
                summary["categorical"][col] = {
                    "unique_count": self.df[col].nunique(),
                    "most_common": (
                        self.df[col].mode().iloc[0]
                        if len(self.df[col].mode()) > 0
                        else None
                    ),
                    "value_counts": self.df[col].value_counts().head(10).to_dict(),
                }

        return summary

    def _detect_outliers(self, numeric_cols):
        outliers = {}
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_count = len(
                self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            )
            outliers[col] = {
                "count": outlier_count,
                "percent": (outlier_count / len(self.df)) * 100,
            }
        return outliers

    def check_consistency(self):
        issues = []
        categorical_cols = self.df.select_dtypes(include=["object"]).columns

        for col in categorical_cols:
            if self.df[col].dtype == "object":
                values = self.df[col].dropna().astype(str)
                if len(values) != len(values.str.lower().drop_duplicates()):
                    issues.append(f"{col}: Mixed case values detected")

                spaces = (
                    self.df[col].str.len() != self.df[col].str.strip().str.len()
                ).sum()
                if spaces > 0:
                    issues.append(
                        f"{col}: {spaces} values with leading/trailing spaces"
                    )

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if any(
                keyword in col.lower()
                for keyword in [
                    "price",
                    "cost",
                    "amount",
                    "quantity",
                    "count",
                    "age",
                    "rating",
                ]
            ):
                negative_count = (self.df[col] < 0).sum()
                if negative_count > 0:
                    issues.append(f"{col}: {negative_count} negative values")

        return issues

    def analyze_correlations(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        high_correlations = []

        if len(numeric_cols) > 1:
            corr_matrix = self.df[numeric_cols].corr()

            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        high_correlations.append(
                            {
                                "col1": corr_matrix.columns[i],
                                "col2": corr_matrix.columns[j],
                                "correlation": corr_val,
                            }
                        )

        return high_correlations

    def check_value_ranges(self):
        range_issues = []
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            col_lower = col.lower()
            min_val, max_val = self.df[col].min(), self.df[col].max()

            if "age" in col_lower and (min_val < 0 or max_val > 120):
                range_issues.append(f"{col}: Age range {min_val}-{max_val} unrealistic")
            elif (
                any(keyword in col_lower for keyword in ["price", "cost"])
                and min_val < 0
            ):
                range_issues.append(f"{col}: Negative prices found")
            elif "rating" in col_lower and (min_val < 0 or max_val > 10):
                range_issues.append(
                    f"{col}: Rating range {min_val}-{max_val} outside 0-10 scale"
                )
            elif any(keyword in col_lower for keyword in ["percent", "rate", "%"]) and (
                min_val < 0 or max_val > 100
            ):
                range_issues.append(f"{col}: Percentage values outside 0-100% range")

        return range_issues

    def calculate_quality_scores(self, consistency_issues, range_issues):
        completeness = (
            1 - self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))
        ) * 100
        uniqueness_score = min(
            (self.df.nunique().sum() / (len(self.df) * len(self.df.columns))) * 100, 100
        )
        consistency_score = max(0, 100 - len(consistency_issues) * 10)
        range_score = max(0, 100 - len(range_issues) * 15)
        overall_score = (
            completeness + uniqueness_score + consistency_score + range_score
        ) / 4

        return {
            "completeness": completeness,
            "uniqueness": uniqueness_score,
            "consistency": consistency_score,
            "range_validity": range_score,
            "overall": overall_score,
        }

    def generate_recommendations(
        self, quality_assessment, consistency_issues, range_issues, quality_scores
    ):
        recommendations = []
        rows = len(self.df)

        if rows < 1000:
            recommendations.append(
                f"Increase dataset size to 1000+ rows (current: {rows})"
            )

        if quality_assessment["duplicate_count"] > 0:
            recommendations.append("Remove duplicate rows")

        if sum(quality_assessment["missing_values"].values()) > 0:
            recommendations.append("Address missing values")

        if consistency_issues:
            recommendations.append("Fix data consistency issues")

        if range_issues:
            recommendations.append("Correct unrealistic value ranges")

        low_variety_cols = [
            col
            for col in self.df.columns
            if self.df[col].nunique() / len(self.df) < 0.1
            and self.df[col].nunique() > 1
        ]
        if low_variety_cols:
            recommendations.append(
                f"Increase variety in: {', '.join(low_variety_cols)}"
            )

        if quality_scores["overall"] < 80:
            recommendations.append("Consider regenerating dataset (quality < 80%)")

        return recommendations

    def run_full_analysis(self):
        basic_info = self.get_basic_info()
        column_info = self.analyze_columns()
        quality_assessment = self.assess_data_quality()
        statistical_summary = self.get_statistical_summary()
        consistency_issues = self.check_consistency()
        correlations = self.analyze_correlations()
        range_issues = self.check_value_ranges()
        quality_scores = self.calculate_quality_scores(consistency_issues, range_issues)
        recommendations = self.generate_recommendations(
            quality_assessment, consistency_issues, range_issues, quality_scores
        )

        return {
            "basic_info": basic_info,
            "column_info": column_info,
            "quality_assessment": quality_assessment,
            "statistical_summary": statistical_summary,
            "consistency_issues": consistency_issues,
            "correlations": correlations,
            "range_issues": range_issues,
            "quality_scores": quality_scores,
            "recommendations": recommendations,
        }
