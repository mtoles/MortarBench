"""
Load three questions of varying difficulty levels from the Excel file.
"""

import pandas as pd


def load_questions():
    """Load three questions of varying difficulty levels."""
    # Load the Excel file
    df = pd.read_excel("data/Labeled Questions and Answers.xlsx")

    # Select three questions of varying difficulty levels
    # Easy: Simple yes/no question about address matching
    question_easy = df.iloc[
        4
    ]  # Row 5: Compare the borrower's address on the account statement with the current address on the loan application

    # Medium: Question requiring ID list identification
    question_medium = df.iloc[
        2
    ]  # Row 3: Identify any large deposits on the borrower's bank statements

    # Hard: Complex question requiring multiple criteria analysis
    question_hard = df.iloc[
        17
    ]  # Row 18: Are there recurring debt payments that are not disclosed on the loan application or appearing on the credit report?

    return question_easy, question_medium, question_hard


if __name__ == "__main__":
    question_easy, question_medium, question_hard = load_questions()

    print("Question Easy (Simple yes/no):")
    print(f"Question: {question_easy['Question']}")
    print(f"Rephrased: {question_easy['Rephrased Question']}")
    print(f"Answer: {question_easy['Revised Answer V2']}")
    print(f"Type: {question_easy['Answer Type']}")
    print()

    print("Question Medium (ID list identification):")
    print(f"Question: {question_medium['Question']}")
    print(f"Rephrased: {question_medium['Rephrased Question']}")
    print(f"Answer: {question_medium['Revised Answer V2']}")
    print(f"Type: {question_medium['Answer Type']}")
    print()

    print("Question Hard (Complex multi-criteria analysis):")
    print(f"Question: {question_hard['Question']}")
    print(f"Rephrased: {question_hard['Rephrased Question']}")
    print(f"Answer: {question_hard['Revised Answer V2']}")
    print(f"Type: {question_hard['Answer Type']}")

