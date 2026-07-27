# SmartHire — Final project report

## Executive summary

SmartHire is a resume-to-job matching and career-guidance engine built entirely
with classical machine learning. A user can upload a PDF resume or paste text,
predict a resume category, search a merged corpus of 67,790 Naukri and LinkedIn
job listings,
view an interpretable fit score, and identify missing skills for a target role.

## Data

The resume dataset contains 9,544 records and 35 fields covering skills,
education, experience, responsibilities, positions, and job requirements. The
processed job corpus contains 67,790 unique listings: 21,346 from Naukri and
46,444 from LinkedIn. Raw, interim, and processed data are separated under
`data/`.

## Methods

Resume classification combines cleaned text from relevant resume fields,
TF-IDF vectorization (maximum 5,000 features), and multinomial logistic
regression. The split is 80/20, stratified, with random state 42.

The recommender builds a TF-IDF index over normalized Naukri and LinkedIn
listings. At
inference, cosine similarity ranks each job against the candidate resume.
The displayed fit score is transparent: 70% text similarity plus 30% detected
skill coverage. It is guidance rather than a calibrated hiring probability.

K-Means clustering with 10 clusters provides unsupervised role-family
discovery. The clustering notebook compares inertia and silhouette behavior
and exposes top terms and positions per cluster.

The skill-gap module detects skills with token-boundary keyword matching,
matches a target role from the resume dataset, and reports present and missing
requirements.

## Evaluation

The resume classifier reaches 86.46% accuracy and 83.67% macro F1 on 1,019
test samples across 19 categories. Eight classes achieve
perfect precision and recall, while rare and seniority-adjacent classes are
harder. The most visible errors are Accountant versus Senior/Staff Accountant,
and Software Engineer versus Software Engineering Manager.

Title-level recommender Precision@5 is 0.276 across 50 reproducible queries.
K-Means selection across k=2–10 chooses k=10 with a silhouette score of 0.0564,
showing that lexical job families overlap substantially. The optional
supervised fit predictor reaches 64.80% accuracy, 60.33% F1, and 0.693 ROC-AUC
on 1,909 held-out samples. Automated tests cover preprocessing, model loading,
classification output, recommendation behavior, skill extraction, evaluation,
and fit-score boundaries.

## Application and architecture

The Streamlit portal has four views: profile classification, job matches,
skill gap, and project details. Expensive model and job-index loading is cached.
PDF text is extracted with pdfplumber. Reusable logic remains under `src`,
while exploration and training remain in numbered notebooks.

## Limitations and responsible use

TF-IDF is lexical and may miss synonyms or contextual meaning. The skill
vocabulary is curated and cannot cover every industry term. Dataset roles and
locations may be stale, and PDF extraction does not OCR scanned images.
Category imbalance affects rare-role recall. SmartHire must not be used as an
automatic hiring gate; its scores should support, not replace, human judgment.

## Conclusion

The project meets the minimum supervised-plus-unsupervised scope: a measured
resume classifier, content-based job recommender, clustering workflow,
skill-gap analysis, automated tests, saved models, reproducible notebooks, and
a working Streamlit demonstration. The interpretable fit score and real job
corpus extend the core scope while retaining classical ML.
