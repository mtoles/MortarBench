# Evaluation Results: gpt-5

**Date:** 2025-12-30 21:30:51

## Configuration Parameters

- **model_id:** gpt-5
- **downsample_size:** None
- **metadata_path:** data/benchmark_dataset_metadata.json
- **test_cases_path:** data/benchmark_dataset_test_cases.jsonl
- **use_domain_expertise:** True
- **excel_path:** data/Labeled Questions and Answers.xlsx
- **skip_preprocessing:** False
- **results_dir:** presentation
- **offset:** 0
- **trials:** 1
- **loan_ids:** None
- **question_col:** Question
- **row_indexes:** None

**Model:** gpt-5

**Dataset Size:** 90

**Overall F1 Accuracy:** 0.870 (87.0%)

**Overall Exact Match Accuracy:** 0.771 (77.1%)

## Metrics for PII==False Subset

**F1 Accuracy (PII==False):** 0.870 (87.0%)

**Exact Match Accuracy (PII==False):** 0.771 (77.1%)

**Dataset Size (PII==False):** 83

## Cost Analysis

**Total Input Tokens:** 1,920,841

**Total Output Tokens:** 337,846

**Input Cost:** $2.40

**Output Cost:** $3.38

**Total Cost:** $5.78

## Accuracy by Answer Type

- **txn_id_list:** Exact Match: 0.737 (73.7%) - 42/57
  F1 Score: 0.869 (86.9%)
- **boolean:** Exact Match: 0.895 (89.5%) - 17/19
  F1 Score: 0.895 (89.5%)
- **account_id_list:** Exact Match: 0.714 (71.4%) - 5/7
  F1 Score: 0.810 (81.0%)

## Sample Errors

### Error 1
**Loan ID:** 86881713506
**Question:** Do the statements reflect any unusually large or irregular deposits requiring documentation?
**Expected:** plaid-2-00037, plaid-2-00049 (txn_id_list)
**Predicted:** ['["plaid-2-00021", "plaid-2-00037", "plaid-2-00048"]']

### Error 2
**Loan ID:** 86881713506
**Question:** Is there an unexplained deposit in a bank's transaction history that could be a sign of unsecured borrowed funds?
**Expected:** plaid-2-00037 (txn_id_list)
**Predicted:** ['["plaid-2-00037", "plaid-2-00021", "plaid-2-00048"]']

### Error 3
**Loan ID:** 86881713506
**Question:** Are there recurring debt payments that are not disclosed on the loan application or appearing on the credit report?
**Expected:** plaid-2-00022, plaid-2-00052, plaid-2-00046, plaid-2-00017, plaid-2-00016, plaid-2-00045, plaid-2-00043, plaid-2-00014, plaid-2-00039, plaid-2-00034 (txn_id_list)
**Predicted:** ['["plaid-2-00017", "plaid-2-00046", "plaid-2-00022", "plaid-2-00052", "plaid-2-00039"]']

### Error 4
**Loan ID:** 86881713506
**Question:** Is there a consistent pattern of deposits on the asset statement that could represent an undisclosed other income source?
**Expected:** plaid-2-00033, plaid-2-00009 (txn_id_list)
**Predicted:** ['["plaid-2-00009", "plaid-2-00033", "plaid-2-00021", "plaid-2-00048", "plaid-2-00015", "plaid-2-00028", "plaid-2-00044", "plaid-2-00057", "plaid-2-00037"]']

### Error 5
**Loan ID:** 84192307554
**Question:** Do the statements reflect any unusually large or irregular deposits requiring documentation?
**Expected:** none (txn_id_list)
**Predicted:** ['["plaid-3-00032", "plaid-3-00008", "plaid-3-00033", "plaid-3-00035", "plaid-3-00011", "plaid-3-00038", "plaid-3-00015", "plaid-3-00019"]']

### Error 6
**Loan ID:** 84192307554
**Question:** Review all asset statements to determine if any account is a joint account. If a joint account is identified, confirm whether all listed account holders are also listed as borrowers on the loan application.
**Expected:** 3434, 7878 (account_id_list)
**Predicted:** ['["1212", "5656", "3434", "7878"]']

### Error 7
**Loan ID:** 84192307554
**Question:** Check the bank account activity for a withdrawal that matches the earnest money deposit (EMD) amount specified in the purchase contract. If a matching transaction is found, provide the transaction date, amount, and any available recipient details to confirm its purpose.
**Expected:** plaid-3-00028 (txn_id_list)
**Predicted:** ['["plaid-3-00028", "plaid-3-00027"]']

### Error 8
**Loan ID:** 84192307554
**Question:** Scan the borrower's bank statements for any payments to creditors that are not listed on either the credit report or the loan application. If found, identify the name of the creditor and the payment amount.
**Expected:** plaid-3-00002, plaid-3-00041, plaid-3-00036 (txn_id_list)
**Predicted:** ['["plaid-3-00001", "plaid-3-00002", "plaid-3-00036", "plaid-3-00041", "plaid-3-00031", "plaid-3-00042"]']

### Error 9
**Loan ID:** 80731120165
**Question:** Is there evidence of rental payment(s) being made from the account?
**Expected:** plaid-4-00004, plaid-4-00017, plaid-4-00060 (txn_id_list)
**Predicted:** ['["plaid-4-00004", "plaid-4-00017"]']

### Error 10
**Loan ID:** 89811904866
**Question:** Do the statements reflect any unusually large or irregular deposits requiring documentation?
**Expected:** none (txn_id_list)
**Predicted:** ['["plaid-5-00001", "plaid-5-00023", "plaid-5-00041", "plaid-5-00009", "plaid-5-00027", "plaid-5-00019", "plaid-5-00037", "plaid-5-00042", "plaid-5-00044", "plaid-5-00045", "plaid-5-00047", "plaid-5-00048"]']

### Error 11
**Loan ID:** 89811904866
**Question:** Do all bank statements include the financial institution name & account number?
**Expected:** no (boolean)
**Predicted:** ['Yes']

### Error 12
**Loan ID:** 89811904866
**Question:** Do the rental income deposits shown on the bank statements align with the gross rental income reported for the property(ies) listed on the loan application (both subject and REO)?
**Expected:** yes (boolean)
**Predicted:** ['No']

### Error 13
**Loan ID:** 83352063666
**Question:** Are there any provided retirement accounts that can be used as a source of funds for the mortgage?
**Expected:** none (account_id_list)
**Predicted:** ['["4411", "5522", "7788", "9921"]']

### Error 14
**Loan ID:** 81613557991
**Question:** Do the statements reflect any unusually large or irregular deposits requiring documentation?
**Expected:** plaid-7-00037 (txn_id_list)
**Predicted:** ['["plaid-7-00021", "plaid-7-00037", "plaid-7-00048"]']

### Error 15
**Loan ID:** 81613557991
**Question:** Are there recurring debt payments that are not disclosed on the loan application or appearing on the credit report?
**Expected:** plaid-7-00014, plaid-7-00043, plaid-7-00016, plaid-7-00045, plaid-7-00017, plaid-7-00046, plaid-7-00022, plaid-7-00052, plaid-7-00034, plaid-7-00039 (txn_id_list)
**Predicted:** ['["plaid-7-00014", "plaid-7-00043", "plaid-7-00016", "plaid-7-00045", "plaid-7-00017", "plaid-7-00046", "plaid-7-00022", "plaid-7-00052"]']

### Error 16
**Loan ID:** 81613557991
**Question:** Is there an unexplained deposit in a bank's transaction history that could be a sign of unsecured borrowed funds?
**Expected:** plaid-7-00037 (txn_id_list)
**Predicted:** ['["plaid-7-00037", "plaid-7-00021", "plaid-7-00048"]']

### Error 17
**Loan ID:** 81613557991
**Question:** Is there a consistent pattern of deposits on the asset statement that could represent an undisclosed other income source?
**Expected:** plaid-7-00009, plaid-7-00033 (txn_id_list)
**Predicted:** ['["plaid-7-00021", "plaid-7-00048", "plaid-7-00009", "plaid-7-00033"]']

### Error 18
**Loan ID:** 81301535410
**Question:** Do the statements reflect any unusually large or irregular deposits requiring documentation?
**Expected:** none (txn_id_list)
**Predicted:** ['["plaid-10-00002", "plaid-10-00010", "plaid-10-00020", "plaid-10-00025", "plaid-10-00029", "plaid-10-00039", "plaid-10-00044", "plaid-10-00045", "plaid-10-00046", "plaid-10-00049", "plaid-10-00047", "plaid-10-00048", "plaid-10-00050", "plaid-10-00051"]']

### Error 19
**Loan ID:** 81301535410
**Question:** Do the rental income deposits shown on the bank statements align with the gross rental income reported for the property(ies) listed on the loan application (both subject and REO)?
**Expected:** plaid-10-00020, plaid-10-00039, plaid-10-00025, plaid-10-00044 (txn_id_list)
**Predicted:** ['["plaid-10-00002", "plaid-10-00020", "plaid-10-00025", "plaid-10-00039", "plaid-10-00044"]']

