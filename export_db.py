import argparse
import sqlite3
from collections import Counter

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db_name",
        type=str,
        default="data/evaluation.db",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="data/eval_annotations.xlsx",
    )
    args = parser.parse_args()

    # database connection
    DATABASE = args.db_name
    DB_CONN = sqlite3.connect(DATABASE, check_same_thread=False)
    DB_CURSOR = DB_CONN.cursor()

    # export the evaluation results as excel
    evaluation_results = pd.read_sql_query("SELECT * from evaluation_record", DB_CONN)
    evaluation_results.to_excel(args.output_name, index=False)


if __name__ == "__main__":
    main()
