
from transformers import BertTokenizer, DataCollatorWithPadding, BertForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import numpy as np
import evaluate
import random
import json


def gen_train():
    input_dir = 'data/pretrain_essays.json'
    with open(input_dir, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            json_line = json.loads(line)
            rating = json_line["rating"]
            # 过滤低分作文
            if rating == 0:
                continue
            content = json_line["content"][1:] # 第一个元素是标题，去除标题
            # 正例
            if len(content) < 3:
                continue
            for i in range(len(content) - 2):
                sent = content[i] + content[i+1] + content[i+2]
                yield {'label': 1, 'text': sent}
            # 负例
            if len(content) < 4:
                continue
            for i in range(len(content) - 2):
                delete_index = random.randint(0, 2)
                while True:
                    insert_index = random.randint(0, len(content)-1)
                    if insert_index not in [i, i+1, i+2]:
                        break
                if delete_index == 0:
                    sent = content[insert_index] + content[i+1] + content[i+2]
                elif delete_index == 1:
                    sent = content[i] + content[insert_index] + content[i+2]
                elif delete_index == 2:
                    sent = content[i] + content[i+1] + content[insert_index]
                yield {'label': 0, 'text': sent}


def gen_dev():
    input_dir = 'data/all_data.json'
    with open(input_dir, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            json_line = json.loads(line)
            sents = json_line['sents']
            content = [''.join(i) for i in sents]
            # score = json_line['score']
            # 正例
            if len(content) < 3:
                continue
            for i in range(len(content) - 2):
                sent = content[i] + content[i+1] + content[i+2]
                yield {'label': 1, 'text': sent}
            # 负例
            if len(content) < 4:
                continue
            for i in range(len(content) - 2):
                delete_index = random.randint(0, 2)
                while True:
                    insert_index = random.randint(0, len(content)-1)
                    if insert_index not in [i, i+1, i+2]:
                        break
                if delete_index == 0:
                    sent = content[insert_index] + content[i+1] + content[i+2]
                elif delete_index == 1:
                    sent = content[i] + content[insert_index] + content[i+2]
                elif delete_index == 2:
                    sent = content[i] + content[i+1] + content[insert_index]
                yield {'label': 0, 'text': sent}


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=256)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


ds_train = Dataset.from_generator(gen_train)
ds_dev = Dataset.from_generator(gen_dev)

tokenizer = BertTokenizer.from_pretrained("pretrained_bert_models/chinese_roberta_wwm_large_ext/")

tokenized_train = ds_train.map(preprocess_function, batched=True)
tokenized_dev = ds_dev.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

accuracy = evaluate.load("accuracy")

id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}


model = BertForSequenceClassification.from_pretrained("pretrained_bert_models/chinese_roberta_wwm_large_ext/",
                                                      num_labels=2, 
                                                      id2label=id2label, 
                                                      label2id=label2id)

training_args = TrainingArguments(
    output_dir="saved_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
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