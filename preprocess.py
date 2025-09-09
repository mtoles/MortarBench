"""


# JSONL Format

{
    "question": <Rephrased Question>,
    "answer": <Revised Answer V2>,
    "bank_statement": <Test Case JSON>,
    "ulad_du": <ULAD DU XML>,
    "answer_type": <Answer Type>
}

Converts the `data/Labeled Questions and Answers.xlsx` and test case docs into a single JSONL file formatted as above.
<Test Case JSON> is the full text of the test case JSON file (e.g. `data/Test Case 1 Docs/test_case_1_bank_statement_solo.json`) corresponding to the integer in the Test Case Number column of the xlsx file.

"""

import pandas as pd
import json
import os


def load_test_case_json(test_case_number):
    """Load the JSON file for a given test case number."""
    json_path = f"data/Test Case {test_case_number} Docs/test_case_{test_case_number}_bank_statement_solo.json"

    if not os.path.exists(json_path):
        print(f"Warning: JSON file not found: {json_path}")
        return None

    try:
        with open(json_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return None


def load_ulad_xml(test_case_number):
    """Load the ULAD DU XML file for a given test case number."""
    xml_path = f"data/Test Case {test_case_number} Docs/ULAD DU.xml"
    with open(xml_path, "r") as f:
        return f.read()


def main():
    # Load the Excel file
    excel_path = "data/Labeled Questions and Answers.xlsx"

    if not os.path.exists(excel_path):
        print(f"Error: Excel file not found: {excel_path}")
        return

    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    print(f"Loaded {len(df)} rows from Excel file")
    print(f"Columns: {list(df.columns)}")

    # Create output JSONL file
    output_path = "data/preprocessed_data.jsonl"

    with open(output_path, "w") as outfile:
        for idx, row in df.iterrows():
            # Extract required columns (adjust column names based on actual Excel structure)
            question = row["Rephrased Question"]
            answer = row["Revised Answer V2"]
            test_case_number = row["Test Case Number"]
            answer_type = row["Answer Type"]
            assert answer_type in [
                "yes",
                "no",
                "none",
                "id_list",
            ], f"Invalid answer type: {answer_type}"

            # Load corresponding JSON file
            bank_statement = load_test_case_json(int(test_case_number))

            # Load corresponding ULAD DU XML file
            ulad_du = load_ulad_xml(int(test_case_number))

            # Create JSONL entry
            entry = {
                "question": question,
                "answer": answer,
                "bank_statement": bank_statement,
                "ulad_du": ulad_du,
                "answer_type": answer_type,
            }

            # Write to JSONL file
            outfile.write(json.dumps(entry) + "\n")

    print(f"Successfully created {output_path}")


if __name__ == "__main__":
    main()
