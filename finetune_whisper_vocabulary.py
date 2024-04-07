from datasets import load_dataset, Audio, ClassLabel
import pandas as pd
import numpy as np
import evaluate
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, TrainingArguments, Trainer
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

speaknow = pd.read_csv("./SpeakNow_test_data.csv")

scoreMap = {
    1: 'Very Poor',
    2: 'Poor',
    3: 'Average',
    4: 'Good',
    5: 'Very Good',
    6: 'Excellent',
}

metadata = pd.DataFrame()
metadata['file_name'] = np.concatenate([
    speaknow['assessment_id'].astype(str) + '-1.mp3',
    speaknow['assessment_id'].astype(str) + '-2.mp3',
    speaknow['assessment_id'].astype(str) + '-3.mp3',
    speaknow['assessment_id'].astype(str) + '-4.mp3',
    speaknow['assessment_id'].astype(str) + '-5.mp3'
], 0)
metadata['label'] = np.concatenate([
    np.round(speaknow['vocab_avg'].astype(float)).astype(int).apply(lambda x: scoreMap[x]),
    np.round(speaknow['vocab_avg'].astype(float)).astype(int).apply(lambda x: scoreMap[x]),
    np.round(speaknow['vocab_avg'].astype(float)).astype(int).apply(lambda x: scoreMap[x]),
    np.round(speaknow['vocab_avg'].astype(float)).astype(int).apply(lambda x: scoreMap[x]),
    np.round(speaknow['vocab_avg'].astype(float)).astype(int).apply(lambda x: scoreMap[x])
], 0)
metadata.to_csv('./data/metadata.csv', index=False)

labels = list(scoreMap.values())
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

model_id = "openai/whisper-medium"
feature_extractor = AutoFeatureExtractor.from_pretrained(
    model_id, do_normalize=True
)

model = AutoModelForAudioClassification.from_pretrained(
    model_id, num_labels=len(label2id), label2id=label2id, id2label=id2label
)

dataset = load_dataset("audiofolder", data_dir="./data", num_proc=6)
dataset = dataset['train'].train_test_split(test_size=0.2, shuffle=False)
dataset = dataset.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))
dataset = dataset.cast_column("label", ClassLabel(num_classes=len(scoreMap.values()), names=list(scoreMap.values()), names_file=None, id=None))

def preprocess_function(data):
    audio_arrays = [x["array"] for x in data["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=int(feature_extractor.sampling_rate * 30),
        truncation=True,
    )
    return inputs

batch_size = 3
model_name = model_id.split("/")[-1]
gradient_accumulation_steps = 1
num_train_epochs = 10

dataset_encoded = dataset.map(
    preprocess_function,
    remove_columns="audio",
    batched=True,
    batch_size=batch_size,
    # num_proc=6,
)

training_args = TrainingArguments(
    f"{model_name}-vocabulary-accuracy",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_train_epochs,
    warmup_ratio=0.1,
    logging_steps=5,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True,
    push_to_hub=False,
)

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

class MseTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)

        err = torch.log(torch.outer(
            torch.exp(labels),
            torch.exp(-torch.arange(6))
        ))
        sqr_err = torch.square(err)
        prob = torch.nn.Softmax(-1)(outputs.logits)
        
        mse_loss = torch.sqrt(torch.sum(prob*sqr_err, -1))
        loss = torch.sum(mse_loss)

        return (loss, outputs) if return_outputs else loss

class OrdinalCrossEntropyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)

        pred_labels = torch.argmax(outputs.logits, -1)
        abs_err = torch.abs(pred_labels - labels)/5 + 1
        ce = torch.nn.CrossEntropyLoss(reduction='none')(outputs.logits, labels)

        oce_loss = abs_err * ce
        loss = torch.sum(oce_loss)

        return (loss, outputs) if return_outputs else loss

trainer = OrdinalCrossEntropyTrainer(
    model,
    training_args,
    train_dataset=dataset_encoded["train"].with_format("torch"),
    eval_dataset=dataset_encoded["test"].with_format("torch"),
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
)

trainer.train()

exit()
