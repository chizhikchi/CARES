from re import X
import pandas as pd 
import ast
from datasets import Dataset, load_dataset
import torch
import copy
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score,
    classification_report
)

def tokenize_function(batch, tokenizer):
    return tokenizer(batch['full_text'], truncation=True)

def prepare_dataset(tokenizer):
    # loading and pre-processing the data
    dataset = load_dataset("chizhikchi/CARES", use_auth_token=True)
    mlb = MultiLabelBinarizer()
    train_labels = mlb.fit_transform(dataset['train']['chapters'])
    test_labels = mlb.fit_transform(dataset['test']['chapters'])
    dataset['train'] = dataset['train'].add_column('int_labels', list(train_labels))
    dataset['test'] = dataset['test'].add_column('int_labels', list(test_labels))
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer), 
        batched=True, 
        remove_columns=['iddoc', 'id', 'full_text', 'icd10', 'general', 'area', 'chapters'])
    tokenized_dataset = tokenized_dataset.map(
        lambda sample: {'labels': [float(i) for i in sample['int_labels']]}, 
        remove_columns=['int_labels']
        )
    return tokenized_dataset

def my_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 3e-5, 5e-5, log=True),
        # "num_train_epochs": trial.suggest_int("num_train_epochs", 30, 60),
        "seed": trial.suggest_int("seed", 320, 327),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32]),
        "per_device_eval_batch_size": trial.suggest_categorical("per_device_eval_batch_size", [16, 32]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-12, 1e-1, log=True),
        "adam_epsilon": trial.suggest_float("adam_epsilon", 1e-10, 1e-6, log=True),
        "warmup_steps": trial.suggest_int('warmup_steps', 0, 1000)
    }

def compute_objective(metrics):
    metrics = copy.deepcopy(metrics)
    macro_f1 = metrics.pop("eval_macro_f1", None)
    loss = metrics.pop("eval_loss", None)
    _ = metrics.pop("epoch", None)
    # Remove speed metrics
    speed_metrics = [m for m in metrics.keys() if m.endswith("_runtime") or m.endswith("_per_second")]
    for sm in speed_metrics:
        _ = metrics.pop(sm, None)
    return loss if len(metrics) == 0 else macro_f1

def compute_metrics(eval_pred):
    """Computes metrics during evaluation.
    
    Returns:
        A dictionary with the name of the metrics as keys and their score as float values"""
    logits, labels = eval_pred
    label_names = ['1', '2', '4', '6', '7', '8', '9', '10', '11', '12', '13', '14', '17', '18', '19', '21']

    sig = torch.nn.Sigmoid()
    logits = sig(torch.from_numpy(logits)).numpy()
    preds = (logits>0.5).astype(np.float32)
    micro_precision = precision_score(y_true=labels, y_pred=preds, average='micro')
    macro_precision = precision_score(y_true=labels, y_pred=preds, average='macro')
    micro_recall = recall_score(y_true=labels, y_pred=preds, average='micro')
    macro_recall = recall_score(y_true=labels, y_pred=preds, average='macro')
    micro_f1 = f1_score(y_true=labels, y_pred=preds, average='micro')
    macro_f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    print('\n', classification_report(labels, preds, target_names=[str(i) for i in label_names], digits=4))
    return{
        'micro_precision': micro_precision,
        'macro_precision': macro_precision,
        'micro_recall': micro_recall,
        'macro_recall': macro_recall,
        'micro_f1': micro_f1,
        'macro_f1': macro_f1
    }
