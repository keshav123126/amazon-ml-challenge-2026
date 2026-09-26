# Amazon ML Challenge 2026 — Business Entity Resolution

A machine learning pipeline developed for the Amazon ML Challenge 2026
Business Entity Resolution problem.

## Project Overview

The goal is to identify all Source2 and Source3 records that correspond
to each Source1 reference business entity.

The solution uses:

- Text normalization
- Multi-key blocking
- SQLite indexing
- RapidFuzz similarity features
- Logistic Regression
- Probability-based matching
- Candidate generation and ranking

## Pipeline

```text
Source1
   |
   v
Text Normalization
   |
   v
Blocking / Candidate Generation
   |
   v
Candidate Pairs
   |
   v
Similarity Features
   |
   v
Logistic Regression
   |
   v
Final Matching
@"
# Amazon ML Challenge 2026 - Business Entity Resolution

## Approach

The solution performs business entity resolution between Source1 reference entities and noisy Source2/Source3 records.

### 1. Text Normalization
Business names and addresses are normalized by:
- converting text to lowercase
- removing accents
- replacing punctuation with spaces
- normalizing whitespace
- creating compact forms without spaces

### 2. Candidate Generation / Blocking
Exact comparison of every Source1 record against all Source2/Source3 records is avoided.

SQLite indexes are used with multiple blocking keys based on:
- country + normalized name prefix
- country + normalized address prefix
- country + first name words

The union of candidates from these blocking strategies forms the candidate set.

### 3. Matching Model
Candidate pairs are evaluated using a Logistic Regression classifier.

Features include:
- exact normalized name match
- compact name match
- exact normalized address match
- compact address match
- RapidFuzz name similarity
- RapidFuzz address similarity
- token-set similarity
- WRatio similarity
- length differences
- prefix agreement

A probability threshold is applied to determine final matches.

### 4. Output
The solution produces:
- matching_results.tsv
- candidate_pairs.tsv

Each Source1 entity appears exactly once in both output files.

## Main Files

- common.py - text normalization utilities
- build_better_index.py - training candidate index
- build_test_index.py - test candidate index
- train_match_model.py - matching model training
- run_inference.py - inference pipeline
- final_inference.py - restart-safe final inference
- check_block_recall.py - blocking recall evaluation
"@ | Set-Content submission\code\business_entity_resolution\README.md