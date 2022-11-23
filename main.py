import argparse
from train import train_model, test_model
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model_name',
    help = 'select the model to train among the following: bio-bert-spanish, roberta-biomedical-clinical, bert-base-spanish'
)
parser.add_argument(
    '--do',
    help = 'specify whether you want to "train" or "test" already an fine-tuned model'
)
args = parser.parse_args()

if args.do == 'train':
    checkpoints = {
        'bio-bert-spanish': "mrojas/bio-bert-base-spanish-wwm-cased",
        'bert-base-spanish': "dccuchile/bert-base-spanish-wwm-cased",
        'roberta-biomedical-clinical': "PlanTL-GOB-ES/roberta-base-biomedical-es"
    }
    checkpoint = checkpoints[args.model_name]
    if checkpoint == "mrojas/bio-bert-base-spanish-wwm-cased":
        hyperparameters = {
            'num_train_epochs': 25,
            'learning_rate': 4.181013432530947e-05, 
            'seed': 321, 
            'per_device_train_batch_size': 16, 
            'per_device_eval_batch_size': 32, 
            'weight_decay': 3.190500833235664e-11, 
            'adam_epsilon': 1.241521755885265e-07, 
            'warmup_steps': 236
        }
    elif checkpoint == "dccuchile/bert-base-spanish-wwm-cased":
        hyperparameters = {
            'num_train_epochs': 34,
            'learning_rate': 4.862043671050906e-05, 
            'seed': 322, 
            'per_device_train_batch_size': 16, 
            'per_device_eval_batch_size': 32, 
            'weight_decay': 1.0026204622214607e-07, 
            'adam_epsilon': 1.874740778707177e-08, 
            'warmup_steps': 805
        }
    elif checkpoint == "PlanTL-GOB-ES/roberta-base-biomedical-es":
        hyperparameters = {
            'num_train_epochs': 25,
            'learning_rate': 4.181013432530947e-05, 
            'seed': 321, 
            'per_device_train_batch_size': 16, 
            'per_device_eval_batch_size': 32, 
            'weight_decay': 3.190500833235664e-11, 
            'adam_epsilon': 1.241521755885265e-07, 
            'warmup_steps': 236
            } 
    train_model(checkpoint, hyperparameters=hyperparameters)
        
elif args.do == 'test':
    checkpoints = {
        'bio-bert-spanish': 'chizhikchi/cares-biobert-base',
        'bert-base-spanish': 'chizhikchi/cares-bert-base',
        'roberta-biomedical-clinical': 'chizhikchi/cares-roberta-clinical'
    }
    checkpoint = checkpoints[args.model_name]
    if checkpoint == 'chizhikchi/cares-biobert-base':
        hyperparameters = {
            'num_train_epochs': 25,
            'learning_rate': 4.181013432530947e-05, 
            'seed': 321, 
            'per_device_train_batch_size': 16, 
            'per_device_eval_batch_size': 32, 
            'weight_decay': 3.190500833235664e-11, 
            'adam_epsilon': 1.241521755885265e-07, 
            'warmup_steps': 236
        }
    elif checkpoint == 'chizhikchi/cares-bert-base':
        hyperparameters = {
            'num_train_epochs': 34,
            'learning_rate': 4.862043671050906e-05, 
            'seed': 322, 
            'per_device_train_batch_size': 16, 
            'per_device_eval_batch_size': 32, 
            'weight_decay': 1.0026204622214607e-07, 
            'adam_epsilon': 1.874740778707177e-08, 
            'warmup_steps': 805
        }
    elif checkpoint == 'chizhikchi/cares-roberta-clinical':
        hyperparameters = {
            'num_train_epochs': 25,
            'learning_rate': 4.181013432530947e-05, 
            'seed': 321, 
            'per_device_train_batch_size': 16, 
            'per_device_eval_batch_size': 32, 
            'weight_decay': 3.190500833235664e-11, 
            'adam_epsilon': 1.241521755885265e-07, 
            'warmup_steps': 236
            } 
    test_model(checkpoint, hyperparameters=hyperparameters)