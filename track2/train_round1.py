from transformers import BertTokenizerFast, DataCollatorForTokenClassification, BertForTokenClassification, TrainingArguments, Trainer
from datasets import Dataset
import numpy as np
import evaluate
import json
import re

label_list = ["O", "I"]
id2label = {0: "O", 1: "I"}
label2id = {"O": 0, "I": 1}


def gen_train():
    with open("data/train.json", 'r') as f:
        data = f.readlines()
        essays = json.loads(data[0])
        for essay in essays:
            if len(essay["Text"]) != len(essay["ParagraphTopic"]):
                print(essay)
                continue
            title = essay["Title"]
            for i in range(len(essay["Text"])):
                text = essay["Text"][i].replace('?', '？')
                labels = [0] * len(text)
                ts = essay["ParagraphTopic"][i].replace('?', '？')
                rex = re.search(ts, text)
                if not rex:
                    rex = re.search(ts[:-1], text)
                if not rex:
                    rex = re.search(ts[1:], text)  
                if not rex:
                    print(ts, text)
                    continue
                span = rex.span()
                for i in range(span[0], span[1]):
                    labels[i] = 1
                labels = [-100] * len(title) + labels
                yield {"ner_tags": labels, 'tokens': list(text), 'title': list(title)}

                
def gen_dev():
    with open("data/val.json", 'r') as f:
        data = f.readlines()
        essays = json.loads(data[0])
        for essay in essays:
            if len(essay["Text"]) != len(essay["ParagraphTopic"]):
                print(essay)
                continue
            title = essay["Title"]
            for i in range(len(essay["Text"])):
                text = essay["Text"][i].replace('?', '？')
                labels = [0] * len(text)
                ts = essay["ParagraphTopic"][i].replace('?', '？')
                rex = re.search(ts, text)
                if not rex:
                    rex = re.search(ts[:-1], text)
                if not rex:
                    rex = re.search(ts[1:], text)  
                span = rex.span()
                for i in range(span[0], span[1]):
                    labels[i] = 1
                labels = [-100] * len(title) + labels
                yield {"ner_tags": labels, 'tokens': list(text), 'title': list(title)}


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)]
    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples['title'], examples["tokens"], 
                                 truncation=True, max_length=512, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs
    
ds_train = Dataset.from_generator(gen_train)
ds_dev = Dataset.from_generator(gen_dev)

tokenizer = BertTokenizerFast.from_pretrained("pretrained_bert_models/chinese_roberta_wwm_large_ext/")

tokenized_train = ds_train.map(tokenize_and_align_labels, batched=True)
tokenized_dev = ds_dev.map(tokenize_and_align_labels, batched=True)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

seqeval = evaluate.load("seqeval")


model = BertForTokenClassification.from_pretrained("pretrained_bert_models/chinese_roberta_wwm_large_ext/",
                                                      num_labels=2, 
                                                      id2label=id2label, 
                                                      label2id=label2id)

training_args = TrainingArguments(
    output_dir="saved_model",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_dev,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()