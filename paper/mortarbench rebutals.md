# Response to Reviewer 65nd

We thank the reviewer for their thoughtful feedback and attention to detail.
We respond accordingly:

**There is a significant lack of narrative and incremental structure throughout the study.**

The reviewer provides helpful feedback.
We originally wrote the paper to adhere to the traditional Introduction/Methods/Results/Discussion format, but in retrospect it appears to have confused multiple reviewers.
We will restructure the paper as suggested.

**The authors do not clearly specify or justify which models they are comparing in the text.**

This is a fair criticism.
Unless otherwise stated, all figures and analysis report Gemini 3.1 Pro, which we chose because it was the strongest performing baseline.
This was never stated explicitly, and the structural problem identified above made it harder still for the reader to infer.
We will state the model and the reason for its selection in each figure caption and at the start of each analysis section.

**There is no information regarding the reasoning behind choosing these specific closed models, and it is unclear why open-source models were not evaluated.
Including prominent open-source baselines, such as Llama 3 or Mistral, would provide a much fairer and more comprehensive evaluation landscape.**

We chose those three models due to their strength in related tasks and their prevalence, to the best of our knowledge, in the US mortgage underwriting industry.
As the reviewer points out, a more comprehensive analysis would be useful.
Therefore we update the results table to include DeepSeek V4 Pro, Kimi K2.6, Qwen3 32B, and Mistral Small 24B.
Reviewers will note that CRIT continues to improve accuracy on 6/7 baseline models with statistically significant increases in 5/7 models.

%   | Method            | F1                   | EM                   | Boolean  | Txn IDs  | Account IDs | FP Qs    | FP/Q    | FN Qs   | FN/Q     | p      |  
%   | \----------------- | \-------------------- | \-------------------- | \-------- | \-------- | \----------- | \-------- | \------- | \------- | \-------- | \------ |  
%   | Claude Sonnet 4.6 | 56.1 (54.6-56.9)     | 51.4 (50.5-52.1)     | 36.3     | 63.0     | 85.0        | 48.7     | 2.5     | 1.7     | 0.05     | \--     |  
%   | \+ CRIT            | 61.6 (60.2-62.9)     | 58.7 (57.1-59.8)     | 36.3     | 72.6     | 85.0        | 34.3     | 1.2     | 2.7     | 0.05     | \<0.001 |  
%   | GPT-5.5           | 79.9 (79.2-80.8)     | 76.8 (76.1-77.1)     | 94.6     | 72.3     | 93.3        | 34.7     | 1.9     | 1.3     | 0.03     | \--     |  
%   | \+ CRIT            | 80.4 (79.4-81.1)     | 77.5 (77.0-78.0)     | 94.6     | 73.1     | 93.3        | 31.0     | 1.2     | 7.7     | 0.14     | 0.31   |  
%   | Gemini 3.1 Pro    | 80.8 (80.0-82.2)     | 77.1 (76.1-78.2)     | \*\*97.0\*\* | 72.2     | \*\*96.7\*\*    | 36.7     | 1.1     | \*\*0.0\*\* | \*\*0.00\*\* | \--     |  
%   | \+ CRIT            | \*\*83.6\*\* (83.1-84.3) | \*\*80.5\*\* (80.1-81.2) | \*\*97.0\*\* | 77.0     | \*\*96.7\*\*    | 29.3     | 1.0     | 3.3     | 0.06     | \<0.001 |  
%   | DeepSeek V4 Pro   | 80.0 (79.1-81.6)     | 77.7 (76.6-79.3)     | 90.5     | 78.2     | 74.4        | 27.0     | 1.9     | 1.3     | 0.03     | \--     |  
%   | \+ CRIT            | 79.3 (78.6-80.1)     | 77.0 (76.2-77.8)     | 90.5     | 77.0     | 74.4        | 26.7     | 1.7     | 4.0     | 0.08     | 0.74   |  
%   | Kimi K2.6         | 67.9 (67.8-67.9)     | 64.4 (63.8-64.9)     | 73.2     | 65.5     | 79.2        | 41.0     | 3.1     | 4.5     | 0.12     | \--     |  
%   | \+ CRIT            | 76.0                 | 73.1                 | 73.2     | \*\*79.6\*\* | 79.2        | \*\*26.0\*\* | 1.1     | 2.0     | 0.05     | \<0.001 |  
%   | Qwen3 32B         | 53.3 (52.9-53.9)     | 48.6 (48.4-48.9)     | 47.0     | 57.7     | 58.3        | 49.7     | 1.6     | 11.3    | 0.16     | \--     |  
%   | \+ CRIT            | 60.8 (60.7-61.0)     | 57.4 (56.0-58.2)     | 47.0     | 70.7     | 58.3        | 27.0     | \*\*0.6\*\* | 14.7    | 0.25     | \<0.001 |  
%   | Mistral Small 24B | 46.0                 | 40.4                 | 35.7     | 53.4     | 44.0        | 59.0     | 2.6     | 11.0    | 0.15     | \--     |  
%   | \+ CRIT            | 49.5 (48.6-50.0)     | 47.0 (46.3-47.9)     | 35.7     | 59.4     | 44.0        | 36.0     | 0.9     | 18.7    | 0.36     | 0.002  |

**It remains completely unknown why the failure analysis focused exclusively on Gemini's outputs rather than analyzing a diverse set of models.**

This concern is reasonable, and we should have justified the choice in the paper rather than leaving it implicit.
We focused on Gemini as it is the strongest baseline model.
Manual review of failures requires subjective interpretation of failures in reasoning traces as well as domain expertise, so is quite time-consuming.
In a small sample analysis of other models, we did not detect a significant distribution shift in failure modes, so for brevity and focus we chose only to report on the failure modes of the strongest model.

**It seems that the models do not require a vast amount of inherent domain information to succeed.
Why did you not experiment with adding more context directly to the prompt to guide how the model should behave with regard to the specific task? Testing a highly detailed system prompt or a basic retrieval-augmented generation (RAG) approach would help clarify if the models' struggles stem from a lack of reasoning ability or simply a lack of instruction.**

We experimented with using a RAG pipeline, with and without query rewriting, on the Fannie Mae selling guide, but surprisingly saw no change or a slight decrease in accuracy.
We suspect this is because the selling guide addresses primarily procedural matters (e.g. lender breach of contract, record keeping) rather than how to handle edge cases (e.g.
should we assume a monthly Venmo payment to a roommate qualifies as a rent payment).
Unfortunately, we are unaware of any public source containing this type of domain knowledge.
We previously omitted these results expecting a lack of interest but will include them at the reviewer’s suggestion.

We additionally thank the reviewer for the additional feedback on formatting and typos.

# Response to Reviewer skf3

We thank the reviewer for their thoughtful feedback and attention to detail.
We respond accordingly:

**After analyzing the prompts in Table 5 (Appendix D), it seems that the reported results for CRIT on boolean and account ID questions (e.g. in Table 2\) use the exact same prompts as the baseline results and CRIT**

This is the correct interpretation.
We indeed reran each model with all question types for completeness.
The variance in accuracy between trials is due to the inherent stochasticity in responses by API models.
In retrospect it is clear that this has caused confusion.
We will update the table to **only** show accuracy in transaction ID questions for CRIT.
We note that whether or not the non-transaction ID questions are rerun, our conclusion that CRIT improves accuracy remains unchanged.


We update the results table as follows:

%   | Method            | F1                   | EM                   | Boolean  | Txn IDs  | Account IDs | FP Qs    | FP/Q    | FN Qs   | FN/Q     | p      |  
%   | \----------------- | \-------------------- | \-------------------- | \-------- | \-------- | \----------- | \-------- | \------- | \------- | \-------- | \------ |  
%   | Claude Sonnet 4.6 | 56.1 (54.6-56.9)     | 51.4 (50.5-52.1)     | 36.3     | 63.0     | 85.0        | 48.7     | 2.5     | 1.7     | 0.05     | \--     |  
%   | \+ CRIT            | 61.6 (60.2-62.9)     | 58.7 (57.1-59.8)     | 36.3     | 72.6     | 85.0        | 34.3     | 1.2     | 2.7     | 0.05     | \<0.001 |  
%   | GPT-5.5           | 79.9 (79.2-80.8)     | 76.8 (76.1-77.1)     | 94.6     | 72.3     | 93.3        | 34.7     | 1.9     | 1.3     | 0.03     | \--     |  
%   | \+ CRIT            | 80.4 (79.4-81.1)     | 77.5 (77.0-78.0)     | 94.6     | 73.1     | 93.3        | 31.0     | 1.2     | 7.7     | 0.14     | 0.31   |  
%   | Gemini 3.1 Pro    | 80.8 (80.0-82.2)     | 77.1 (76.1-78.2)     | \*\*97.0\*\* | 72.2     | \*\*96.7\*\*    | 36.7     | 1.1     | \*\*0.0\*\* | \*\*0.00\*\* | \--     |  
%   | \+ CRIT            | \*\*83.6\*\* (83.1-84.3) | \*\*80.5\*\* (80.1-81.2) | \*\*97.0\*\* | 77.0     | \*\*96.7\*\*    | 29.3     | 1.0     | 3.3     | 0.06     | \<0.001 |  
%   | DeepSeek V4 Pro   | 80.0 (79.1-81.6)     | 77.7 (76.6-79.3)     | 90.5     | 78.2     | 74.4        | 27.0     | 1.9     | 1.3     | 0.03     | \--     |  
%   | \+ CRIT            | 79.3 (78.6-80.1)     | 77.0 (76.2-77.8)     | 90.5     | 77.0     | 74.4        | 26.7     | 1.7     | 4.0     | 0.08     | 0.74   |  
%   | Kimi K2.6         | 67.9 (67.8-67.9)     | 64.4 (63.8-64.9)     | 73.2     | 65.5     | 79.2        | 41.0     | 3.1     | 4.5     | 0.12     | \--     |  
%   | \+ CRIT            | 76.0                 | 73.1                 | 73.2     | \*\*79.6\*\* | 79.2        | \*\*26.0\*\* | 1.1     | 2.0     | 0.05     | \<0.001 |  
%   | Qwen3 32B         | 53.3 (52.9-53.9)     | 48.6 (48.4-48.9)     | 47.0     | 57.7     | 58.3        | 49.7     | 1.6     | 11.3    | 0.16     | \--     |  
%   | \+ CRIT            | 60.8 (60.7-61.0)     | 57.4 (56.0-58.2)     | 47.0     | 70.7     | 58.3        | 27.0     | \*\*0.6\*\* | 14.7    | 0.25     | \<0.001 |  
%   | Mistral Small 24B | 46.0                 | 40.4                 | 35.7     | 53.4     | 44.0        | 59.0     | 2.6     | 11.0    | 0.15     | \--     |  
%   | \+ CRIT            | 49.5 (48.6-50.0)     | 47.0 (46.3-47.9)     | 35.7     | 59.4     | 44.0        | 36.0     | 0.9     | 18.7    | 0.36     | 0.002  |

**According to Table 2 the number of false negatives increases by more than 6 for GPT5.5, more than the number of false positives decreases.**

This is an accurate reading of the results.
This appears to be a result of the threshold cutoff of 5, chosen for consistency across models and to avoid accidental p-hacking.
Drawing the reviewer’s attention to Figure 3, we see that a more optimal threshold for GPT-5.5 would in fact be 4, after which we see the largest increase in false negatives.
This illustrates the secondary benefits of our approach with respect to being steerable and model-agnostic.

**The related work section is rather limited… it would be good to read what prior work has tried, with what kind of data and models and how their findings impact your work.**

We thank the reviewer for this suggestion.
Given more space, we would update the related works as follows:

Prior work has explored financial regulatory question answering (QA), including Sohn et al. (2021) and Chen et al. (2024). **These works focus on public-facing questions (e.g. FAQ, textbooks) rather than private, in-production systems.** Work on financial numeric and table QA includes Zhu et al. (2021); Chen et al. (2021); Reddy et al. (2024); Islam et al. (2023).   
**Trivedi et al. (2024) introduces a dataset for bank statement table detection and structure recognition but without a QA component.** Choi et al. (2025) analyzes retrieval-augmented generation (RAG) QA. Loukas et al. (2022) studies entity recognition in financial reports.
Mollaev et al. (2025) releases large-scale anonymized transaction, geo-position, and technical support chat data from a major bank.
In contrast, we contribute a benchmark in the mortgage origination domain focused on the chatbot assistant role.
**To our knowledge, this is the first public dataset for QA directly on bank statement natural language and numerical contents.** Numerous works have explored the ability of LLMs to self-calibrate confidence in their own answers, including Manakul et al. (2023); Kapoor et al. (2024); Kuhn et al. (2023); Kadavath et al. (2022). **In particular, we draw on Zhu et al.’s (2023) recognition that large models can accurately estimate their own confidence;**

**Section 6.2 is difficult to follow.**

We hope to clarify as follows.
Because we are creating synthetic data, we can create multiple versions of the same question, differing only in their documents.
This gives us an opportunity to explore the nature of model errors.
If models make false assumptions about the meaning of a question, we expect answers to be consistently wrong (correlated).
However, if models are properly stochastic, then we expect answers to be wrong at random times (uncorrelated).
To analyze this, we generate two versions of each question (our two sets, orthogonal to the 3 trials in the caption of Table 2).
We then calculate whether answering incorrectly on one question (1st wrong) correlates with answering incorrectly on the other question (2nd wrong).
The four cases present in section 3.5 are:

1. Document set A, negative answer  
2. Document set B, negative answer  
3. Document set C, positive answer  
4. Document set D, positive answer

In section 6.2, we compare whether incorrectness on case (1) predicts incorrectness on case (2) and whether incorrectness on case (3) predicts incorrectness on case (4). 

# Response to Reviewer hwLi

We thank the reviewer for their thoughtful feedback and attention to detail.
We respond accordingly:

**The benchmark is interesting, but its realism is not sufficiently validated.
The paper does not show whether mortgage experts reviewed the generated documents or whether the synthetic data truly matches real-world applications.**

We thank the reviewer for drawing attention to this omission.
We wish to add the following details on how the dataset is, by design, representative.
Our dataset contains two types of documents: bank statements and ULADs.
Bank statements contain the following fields for each transaction:

Date  
Description  
Amount  
Balance

We ensure realism in date for transactions such as payroll with deterministic rules, such as rent exactly once per month.
For amount, we sample from a uniform distribution between the median and 90th percentile.
These were the only summary data available to us at the time.
We choose a uniform distribution to avoid unreasonably large or negative values that would be generated by unbounded (e.g. gaussian) distributions.
Therefore, by definition, the amounts present match real-world distributions to the extent possible given the data available.
Balance is a running sum of transaction amounts, seeded with an initial value.
We feel it is neither necessary nor possible to manually review these numeric fields.

To generate transaction descriptions, we reviewed our personal bank statements and online sources, creating descriptions based on these.
Similar to the question generation process, we remove personally identifiable information, such as ATM addresses and account IDs.
We try to cover the major players and description formats for each category of transaction.
For example, “buy now, pay later” descriptions reflect payments to the following companies: Klarna, Afterpay, Affirm, Sezzle, Zip Co, PayPal in 4\.
SMEs were involved in this process.

The ULAD is a standardized data specification rather than a visual document.
Although it is commonly represented in MISMO XML, the same information can be represented in other structured formats such as JSON.
The majority of fields relevant to questions in our dataset are numeric.
Of these, most are directly computable from bank statements, which we do programmatically.
A few non-numeric fields, such as employer name, are also programmatically guaranteed to match those in the bank statement (sampled from a fixed list of large US companies).
Similarly, fields such as alimony, social security, and child support fields are computed from the bank statements.
Others, such as bank account numbers, are generated randomly.
In summary, the ULAD essentially a format-agnostic summary of applicant financials where relevant values can be derived deterministically from the bank statement.
As a result, we do not believe human review is necessary.

**CRIT changes several parts of the pipeline at once: the reasoning prompt, output format, confidence scoring, and filtering.
Without ablations, it is unclear which component causes the improvement.**

We wish to clarify several details about the CRIT pipeline and why it would not be possible to conduct these ablations:

* There are few axes on which we could explore modifying the reasoning prompt without conducting full-blown manual or automatic prompt engineering, which we consider out of scope for this paper.
We attempted to introduce domain knowledge through RAG but we omitted it because it did not improve scores.
The prompt includes little more than instructions on how to interact with rest of the pipeline; removing these would cause the pipeline to universally fail.
 
* The output format remains unchanged as a JSON-formatted list.
The only difference between CRIT and baseline is the existence of an additional “confidence” field.
Removing this field would break the pipeline.
 
* We are not clear how to ablate scoring without fundamentally breaking the CRIT pipeline.
Because scoring is grounded in the number of assumptions being made at the transaction level, as opposed to some arbitrary sense of parameterized confidence, we argue that adjusting the range of permitted confidence scores would amount to merely a translation on T, with identical discriminative power.
 
* We analyze filtering at threshold levels T \\in \[1, 5\] in Figure 3\.
Ablating filtering entirely is identical to setting T=0

In summary, the individual components are too tightly coupled to ablate individually.


**The paper attributes failures to missing domain knowledge without testing stronger instructions or retrieval**

Following LLM reasoning traces, we can confidently conclude that the reasoning errors are due to lack of domain knowledge because the LLM makes false, domain-specific statements to justify its decisions.
We experimented with using a RAG pipeline, with and without query rewriting, on the Fannie Mae selling guide, but saw no change or a slight decrease in accuracy.
We suspect this is because the selling guide addresses primarily procedural matters (e.g. lender breach of contract, record keeping) rather than how to handle edge cases (e.g. should we assume a monthly Venmo payment to a roommate qualifies as a rent payment).
Unfortunately, we are unaware of any public source containing this type of domain knowledge.
Thus, we conclude that domain knowledge is indeed a limitation, but lacking any textual representation of that domain knowledge, augmenting it remains outside the scope of this study.

Similarly, we intentionally provide limited instructions for the LLM.
As in the case of domain knowledge, many hypothetical improvements may exist to improve accuracy on this task, such as automatic prompt engineering and few shot learning.
However, we feel that the inclusion or exclusion of results for these strategies would not impact the two main claims in this paper: a novel benchmark and a method for improving accuracy for an economically important task.

**CRIT’s apparent bias reduction may simply result from producing fewer positive predictions overall.**

To address the reviewer’s concerns about reduction in overall positive predictions, we analyze the false negative rate on transactions in the name and IVTS experiments.
The results show that in the name experiment, across all models, the false negative rate decreases slightly, rather than decreasing positive predictions overall as the reviewer suggests.
In the IVTS experiment, we see an increase in false negatives, but only in weak models.
This indicates that for large models, the bias mitigation is conclusively **not** driven by a uniform decrease in positive predictions.

Model	Name base	Name CRIT	IVTS base	IVTS CRIT  
Claude 4.6	0.0% (0/9)	0.0% (0/9)	0.0% (0/36)	0.0% (0/36)  
GPT-5.5	0.0% (0/9)	0.0% (0/9)	0.0% (0/36)	0.0% (0/36)  
Gemini 3.1	0.0% (0/9)	0.0% (0/9)	0.0% (0/36)	0.0% (0/36)  
DeepSeek V4	0.0% (0/9)	0.0% (0/9)	0.0% (0/36)	0.0% (0/36)  
Kimi K2.6	0.0% (0/24)	8.3% (1/12)	0.0% (0/96)	0.0% (0/48)  
Qwen3 32B	0.0% (0/9)	0.0% (0/9)	5.6% (2/36)	25.0% (9/36)  
Mistral 24B	16.7% (3/18)	0.0% (0/18)	0.0% (0/72)	38.9% (28/72)  
ALL		3.4% (3/87)	1.3% (1/75)	0.6% (2/348)	12.3% (37/300)

