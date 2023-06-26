## Track 1. Coherence Evaluation

| Team Name | Precision | Recall | Macro-F1 | Accuracy |
| --- | --- | --- | --- | --- |
| EssayFlow | 38.50 | 43.54 | 32.54 | 43.99 |
| Evay Info AI Team | 35.64 | 35.70 | 35.61 | 36.05 |
| <strong>ouchnai (us)</strong> | 36.38 | 41.32 | 33.22 | 34.92 |
| CLsuper | 34.13 | 34.28 | 32.80 | 32.88 |

The Ouchnai solution for Track 1:

Our coherence scoring model encompasses a regression model with two feature extractors: a local coherence discriminative model (LCD) and a punctuation correction model (PC).

The LCD is a sequence classification model fine-tuned on BERT with external data. This model accepts three consecutive sentences as input and generates a probability estimate of the coherence for the sequence. We divided the essay into consecutive sentences, took each as the model input, and obtained the ratio of coherent sequences to total sequences. The PC is a token classification model also fine-tuned on BERT with external data. This model examines the essay's punctuation usage, focusing explicitly on identifying redundant, missing, and misused commas and periods.

We employ a GBRT to map features extracted from LCD and PC into a final global coherence score. We imposed linguistically-informed monotonicity constraints on all features and showed that the regulations enhanced the model's generalization ability.

## Track 2. Text Topic Extraction

| Team Name | Paragraph Accuracy | Full Accuracy | Final Accuracy | Paragraph Similarity | Full Similarity |
| --- | --- | --- | --- | --- | --- |
| wuwuwu | 61.27 | 34.92 | 42.82 | 87.34 | 80.37 |
| <strong>ouchnai (us)</strong> | 62.61 | 33.33 | 42.12 | 85.20 | 79.16 |

The Ouchnai solution for Track 2:

In our approach, we employ two token classification models to identify both paragraph-level and overall theme sentences. 

The first model accepts the essay title connected to a paragraph as input. For each token, it outputs a label indicating whether the token belongs to the theme sentences of the paragraph (designated as a key token). The theme sentences of each paragraph are determined by the ratio of key tokens to the total number of tokens within the sentence. We select the sentence with the highest ratio as the theme sentence for that paragraph. The model is fine-tuned on BERT.

The second model is similar to the first, but the input is a sequence that connects the essay title to all paragraph's theme sentences. We assume that the overall theme sentence is one of the paragraph theme sentences and determine it by calculating the ratio of key tokens to the total number of tokens within each paragraph theme sentence. We select the sentence with the highest ratio as the overall theme sentence. The second model is fine-tuned on the first model.

## Track 3. Paragraph Logical Relation Recognition

| Team Name | Precision | Recall | Macro-F1 | Accuracy |
| --- | --- | --- | --- | --- |
| <strong>ouchnai (us)</strong> | 54.66 | 52.45 | 52.16 | 71.03 |
| wuwuwu | 29.26 | 28.98 | 28.77 | 46.97 |
| Lrt123 | 28.19 | 30.26 | 27.54 | 48.81 |
| BLCU_teamworkers | 27.17 | 27.65 | 25.95 | 48.73 |

The Ouchnai solution for Track 3:

Our approach regards the paragraph-level logical relation recognition task as a sequence classification problem. Specifically, we process a pair of paragraphs as input, and the model determines the logical relationship between these paragraphs. Considering the similarity between this task and sentence-level logical relation recognition, we chose to fine-tune the model trained for Task 4.  

## Track 4. Sentence Logical Relation Recognition

| Team Name | Precision | Recall | Macro-F1 | Accuracy |
| --- | --- | --- | --- | --- |
| <strong>ouchnai (us)</strong> | 36.63 | 36.36 | 34.38 | 53.95 |
| wuwuwu | 23.49 | 25.37 | 23.67 | 39.94 |
| BLCU_teamworkers | 7.55 | 6.30 | 6.32 | 18.35 |

The Ouchnai solution for Track 4:

For Task 4, a sentence-level logical relation recognition task, we adopt a two-stage training strategy for the model. In the initial phase, we utilize an external dataset, TED-CDB, to pre-train a sequence classification model based on BERT. In the subsequent phase, we fine-tune the pre-trained model on the current dataset to enhance its performance for the given task. 
