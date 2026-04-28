#!/usr/bin/env bash
#
# End-to-end script for generating the test_cases_unique folder.
#
# This script:
#   1. Generates base bank statement + ULAD templates via dataset_generator.py
#   2. Copies the generated files to the expected template paths
#   3. Runs generate_test_cases.py to produce mutated test cases from
#      data/questions_unique_generated.csv
#
# Usage:
#   ./run_generate_test_cases.sh                  # default settings
#   ./run_generate_test_cases.sh --skip-generate  # skip step 1, reuse existing templates
#
# Prerequisites:
#   pip install pandas
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Defaults ──────────────────────────────────────────────────────────────
SKIP_GENERATE=false
SEED=""
QUESTIONS="data/questions.csv"
OUTPUT_DIR="generated_data/test_cases_unique"
GEN_DIR="generated_data"
BANK_STMT="$GEN_DIR/bank_statement.json"
BANK_STMT_2="$GEN_DIR/bank_statement_2.json"
ULAD="$GEN_DIR/ulad.json"
ULAD_2="$GEN_DIR/ulad_2.json"

# ── Parse arguments ───────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-generate) SKIP_GENERATE=true; shift ;;
        --seed)          SEED="$2"; shift 2 ;;
        --questions)     QUESTIONS="$2"; shift 2 ;;
        --output)        OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--skip-generate] [--seed SEED] [--questions CSV] [--output DIR]"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo "  Test Case Generation Pipeline"
echo "============================================================"

# ── Step 1: Generate base templates ───────────────────────────────────────
if [ "$SKIP_GENERATE" = false ]; then
    echo ""
    echo "Step 1: Generating base bank statement and ULAD templates..."

    # Build seed flags for dataset_generator.py
    SEED_FLAG=""
    if [ -n "$SEED" ]; then
        SEED_FLAG="--seed $SEED"
    fi

    # Generate primary borrower data
    python dataset_generator.py -n 1 -o "$GEN_DIR" $SEED_FLAG
    # Find the most recently created plaid/ulad files and copy to standard paths
    LATEST_PLAID=$(ls -t "$GEN_DIR"/plaid_*.json 2>/dev/null | head -1)
    LATEST_ULAD=$(ls -t "$GEN_DIR"/ulad_*.json 2>/dev/null | head -1)

    if [ -z "$LATEST_PLAID" ] || [ -z "$LATEST_ULAD" ]; then
        echo "ERROR: dataset_generator.py did not produce expected output files."
        exit 1
    fi

    cp "$LATEST_PLAID" "$BANK_STMT"
    cp "$LATEST_ULAD" "$ULAD"
    echo "  -> Bank statement: $BANK_STMT (from $LATEST_PLAID)"
    echo "  -> ULAD:           $ULAD (from $LATEST_ULAD)"

    # Generate second borrower data (for two-borrower scenarios)
    python dataset_generator.py -n 1 -o "$GEN_DIR" $SEED_FLAG
    LATEST_PLAID_2=$(ls -t "$GEN_DIR"/plaid_*.json 2>/dev/null | head -1)
    LATEST_ULAD_2=$(ls -t "$GEN_DIR"/ulad_*.json 2>/dev/null | head -1)

    cp "$LATEST_PLAID_2" "$BANK_STMT_2"
    cp "$LATEST_ULAD_2" "$ULAD_2"
    echo "  -> Bank statement 2: $BANK_STMT_2 (from $LATEST_PLAID_2)"
    echo "  -> ULAD 2:           $ULAD_2 (from $LATEST_ULAD_2)"
else
    echo ""
    echo "Step 1: Skipped (--skip-generate). Using existing templates."
    # Verify template files exist
    for f in "$BANK_STMT" "$ULAD"; do
        if [ ! -f "$f" ]; then
            echo "ERROR: Required template file not found: $f"
            echo "Run without --skip-generate to create them."
            exit 1
        fi
    done
fi

# ── Step 2: Generate test cases ───────────────────────────────────────────
echo ""
echo "Step 2: Generating mutated test cases..."
echo "  Questions: $QUESTIONS"
echo "  Output:    $OUTPUT_DIR"

SEED_FLAG_TC=""
if [ -n "$SEED" ]; then
    SEED_FLAG_TC="--seed $SEED"
fi

python generate_test_cases.py \
    --questions "$QUESTIONS" \
    --bank-statement "$BANK_STMT" \
    --ulad "$ULAD" \
    --bank-statement-2 "$BANK_STMT_2" \
    --ulad-2 "$ULAD_2" \
    --output "$OUTPUT_DIR" \
    $SEED_FLAG_TC

echo ""
echo "============================================================"
echo "  Done! Test cases written to: $OUTPUT_DIR"
echo "============================================================"
