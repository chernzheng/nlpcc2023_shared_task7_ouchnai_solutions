## Track 1. Coherence Evaluation

| Team Name | Precision | Recall | Macro-F1 | Accuracy |
| --- | --- | --- | --- | --- |
| EssayFlow | 38.50 | 43.54 | 32.54 | 43.99 |
| Evay Info AI Team | 35.64 | 35.70 | 35.61 | 36.05 |
| ouchnai (us) | 36.38 | 41.32 | 33.22 | 34.92 |
| CLsuper | 34.13 | 34.28 | 32.80 | 32.88 |

The Ouchnai solution for Track 1:

Our coherence scoring model encompasses a regression model with two feature extractors: a local coherence discriminative model (LCD) and a punctuation correction model (PC).

The LCD is a sequence classification model fine-tuned on BERT with external data. This model accepts three consecutive sentences as input and generates a probability estimate of the coherence for the sequence. We divided the essay into consecutive sentences, took each as the model input, and obtained the ratio of coherent sequences to total sequences. The PC is a token classification model also fine-tuned on BERT with external data. This model examines the essay's punctuation usage, focusing explicitly on identifying redundant, missing, and misused commas and periods.

We employ a GBRT to map features extracted from LCD and PC into a final global coherence score. We impose linguistically-informed monotonicity constraints on all features, thereby enhancing the model's generalization ability.

## Track 2. Text Topic Extraction

| Team Name | Paragraph Accuracy | Full Accuracy | Final Accuracy | Paragraph Similarity | Full Similarity |
| --- | --- | --- | --- | --- | --- |
| wuwuwu | 61.27 | 34.92 | 42.82 | 87.34 | 80.37 |
| ouchnai (us) | 62.61 | 33.33 | 42.12 | 85.20 | 79.16 |

## Track 3. Paragraph Logical Relation Recognition

| Team Name | Precision | Recall | Macro-F1 | Accuracy |
| --- | --- | --- | --- | --- |
| ouchnai (us) | 54.66 | 52.45 | 52.16 | 71.03 |
| wuwuwu | 29.26 | 28.98 | 28.77 | 46.97 |
| Lrt123 | 28.19 | 30.26 | 27.54 | 48.81 |
| BLCU_teamworkers | 27.17 | 27.65 | 25.95 | 48.73 |

## Track 4. Sentence Logical Relation Recognition

| Team Name | Precision | Recall | Macro-F1 | Accuracy |
| --- | --- | --- | --- | --- |
| ouchnai (us) | 36.63 | 36.36 | 34.38 | 53.95 |
| wuwuwu | 23.49 | 25.37 | 23.67 | 39.94 |
| BLCU_teamworkers | 7.55 | 6.30 | 6.32 | 18.35 |
