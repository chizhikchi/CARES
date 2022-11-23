import pandas as pd 
import ast
from datasets import Dataset
import torch
import numpy as np
from transformers import (
    RobertaTokenizerFast, 
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    TrainingArguments,
    Trainer, 
    DataCollatorWithPadding,
    EarlyStoppingCallback
    )
from utils import (
    prepare_dataset,
    compute_metrics
)

def train_model(checkpoint, hyperparameters):
    
    print(f'\n================== Starting training model {checkpoint} ==================')
    # activating the GPU 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gpu = torch.cuda.device_count()
    if n_gpu == 0:
        raise SystemError('GPU device not found')
    print('\nFound GPU at: {}'.format(torch.cuda.get_device_name(0)))
    
    # data pre-processing
    if checkpoint == 'PlanTL-GOB-ES/roberta-base-biomedical-es':
        tokenizer = RobertaTokenizerFast.from_pretrained(checkpoint)
        steps = 71
    else: 
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        steps = 141
    
    dataset = prepare_dataset(tokenizer)
    print('\nTraining and test datasets ready')
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=len(dataset['train']['labels'][0]))
    model.config.problem_type = 'multi_label_classification'

    training_args = TrainingArguments(
        output_dir=f'CARES/checkpoints/{checkpoint[7:14]}',
        learning_rate=hyperparameters['learning_rate'],
        per_device_train_batch_size=hyperparameters['per_device_train_batch_size'],
        per_device_eval_batch_size=hyperparameters['per_device_eval_batch_size'],
        seed=hyperparameters['seed'],
        weight_decay=hyperparameters['weight_decay'],
        adam_epsilon=hyperparameters['adam_epsilon'],
        warmup_steps=hyperparameters['warmup_steps'],
        evaluation_strategy='steps',
        eval_steps=steps,
        save_steps=steps,
        metric_for_best_model='macro_f1',
        save_total_limit=3,
        load_best_model_at_end=True,
        disable_tqdm=True # set to False 
    )
    
    t = Trainer(
        model,
        training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    t.train()

def test_model(checkpoint, hyperparameters):
    
    print(f'\n================== Starting evaluation of model {checkpoint} ==================')
    # activating the GPU 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gpu = torch.cuda.device_count()
    if n_gpu == 0:
        raise SystemError('GPU device not found')
    print('\nFound GPU at: {}'.format(torch.cuda.get_device_name(0)))

    # data preparation
    if checkpoint == 'chizhikchi/cares-roberta-clinical':
        tokenizer = RobertaTokenizerFast.from_pretrained(checkpoint)
        steps = 71
    else: 
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        steps = 141
    dataset = prepare_dataset(tokenizer)
    test_dataset = dataset['test']
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    #initializing the model
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=len(test_dataset['labels'][0]))
    model.config.problem_type = 'multi_label_classification'
    
    #initializing Trainer
    training_args = TrainingArguments(
        output_dir=f'CARES/checkpoints/{checkpoint[7:14]}',
        learning_rate=hyperparameters['learning_rate'],
        per_device_train_batch_size=hyperparameters['per_device_train_batch_size'],
        per_device_eval_batch_size=hyperparameters['per_device_eval_batch_size'],
        seed=hyperparameters['seed'],
        weight_decay=hyperparameters['weight_decay'],
        adam_epsilon=hyperparameters['adam_epsilon'],
        warmup_steps=hyperparameters['warmup_steps'],
        eval_steps=steps,
        save_steps=steps,
        save_total_limit=1,
        disable_tqdm=False # set to False if you don't want to see progress bars
    )

    t = Trainer(
        model,
        training_args,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    t.evaluate()