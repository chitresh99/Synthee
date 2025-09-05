class DatasetConfig:
    def rows(self, max_rows, min_rows):
        self.max_rows = max_rows
        self.min_rows = min_rows

    def columns(self, max_columns, min_columns):
        self.max_columns = max_columns
        self.min_columns = min_columns


config = DatasetConfig()
# config.rows(max_rows=1000, min_rows=50)
# config.columns(max_columns=8, min_columns=2)
