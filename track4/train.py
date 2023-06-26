
from transformers import BertTokenizer, DataCollatorWithPadding, BertForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import numpy as np
import evaluate
import json


label2id = {
    '并列关系': 0,
    '顺承关系': 1,
    '递进关系': 2,
    '对比关系': 3,
    '让步关系': 4,
    '转折关系': 5,
    '泛化关系': 6,
    '细化关系': 7,
    '客观因果关系': 8,
    '背景关系': 9,
    '特定条件关系': 10,
    '假设条件关系': 11,
    '主观推论关系': 12
}

id2label = {v:k for k, v in label2id.items()}


def gen_train():
    with open("data/train.json", 'r') as f:
        data = f.readlines()
        essays = json.loads(data[0])
        for essay in essays:
            sentence1 = essay['sentence1']['Text']
            sentence2 = essay['sentence2']['Text']
            Relation = essay['Relation']
            yield {'label': label2id[Relation], 'text': sentence1, 'text_pair': sentence2}


def gen_dev():
    with open("data/dev.json", 'r') as f:
        data = f.readlines()
        essays = json.loads(data[0])
        for essay in essays:
            sentence1 = essay['sentence1']['Text']
            sentence2 = essay['sentence2']['Text']
            Relation = essay['Relation']
            yield {'label': label2id[Relation], 'text': sentence1, 'text_pair': sentence2}


def preprocess_function(examples):
    return tokenizer(text=examples["text"], 
                     text_pair=examples['text_pair'],
                     add_special_tokens=True,
                     truncation=True, 
                     max_length=512)


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

model = BertForSequenceClassification.from_pretrained("pretrained_bert_models/chinese_roberta_wwm_large_ext/",
                                                      num_labels=len(id2label), 
                                                      id2label=id2label, 
                                                      label2id=label2id)

training_args = TrainingArguments(
    output_dir="saved_model",
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=5,
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