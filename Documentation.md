# Amazon ML Challenge 2026
# Business Entity Resolution

## 1. Methodology

The solution performs entity resolution between a clean reference
dataset (Source1) and noisy business records from Source2 and Source3.

The pipeline consists of:

1. Text normalization
2. Candidate generation using blocking
3. Feature extraction
4. Machine learning based candidate matching
5. Final threshold-based matching
6. Generation of the required submission files

---

## 2. Data Preprocessing

Business names and addresses are normalized before comparison.

The preprocessing includes:

- Lowercase conversion
- Unicode normalization
- Accent removal
- Punctuation removal
- Whitespace normalization
- Compact representations without spaces

This helps handle variations such as punctuation differences,
capitalization differences and formatting changes.

---

## 3. Blocking / Candidate Generation

Comparing every Source1 record with every Source2 and Source3 record
would require an extremely large number of comparisons.

Therefore, blocking is used to generate candidate pairs.

The candidate generation process uses multiple keys based on:

- Country
- Business name prefix
- Address prefix
- First words of the business name

SQLite indexes are used to efficiently retrieve candidate records.

Candidates generated from different blocking strategies are combined
and duplicates are removed.

The final candidate set is stored in:

`candidate_pairs.tsv`

---

## 4. Matching Model

A Logistic Regression classifier is used to determine whether a
candidate Source2/Source3 record matches a Source1 entity.

The model uses features derived from business names and addresses.

Important features include:

- Exact normalized name match
- Exact compact name match
- Exact normalized address match
- Exact compact address match
- Name fuzzy similarity
- Address fuzzy similarity
- Token-set similarity
- WRatio similarity
- Name length difference
- Address length difference
- Prefix agreement

---

## 5. Model Training

Training candidate pairs are generated using the same blocking strategy.

Positive examples are obtained from the provided ground-truth
relationships.

Negative examples are generated from candidate records that do not
correspond to the Source1 entity.

A Logistic Regression classifier with balanced class weights is trained
on these features.

---

## 6. Final Matching

For each test Source1 entity:

1. Candidate records are generated using blocking.
2. Candidate features are calculated.
3. The trained classifier predicts a match probability.
4. Candidates above the selected probability threshold are retained.
5. Duplicate matched IDs are removed.

The final result is stored in:

`matching_results.tsv`

---

## 7. Output Files

### matching_results.tsv

Contains one row for every Source1 test entity.

Columns:

- `source1_entity_id`
- `matched_entity_ids`

Multiple matched Source2/Source3 IDs are represented as
comma-separated values.

An empty value represents no predicted match.

### candidate_pairs.tsv

Contains one row for every Source1 test entity.

Columns:

- `source1_entity_id`
- `candidate_entity_ids`

Multiple candidate IDs are represented as comma-separated values.

---

## 8. Experiments

Blocking recall was evaluated on a sample of the training data to verify
that the blocking strategy could retrieve true matching records.

A Logistic Regression model was evaluated using different probability
thresholds.

The selected threshold was 0.70, balancing precision and recall while
considering the precision-heavy F0.5 evaluation metric.

---

## 9. Conclusion

The final pipeline combines efficient blocking with fuzzy string
similarity features and supervised machine learning.

Blocking substantially reduces the number of comparisons required while
the matching model provides a probability-based decision for candidate
pairs.