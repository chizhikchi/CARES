import argparse
from train import train_model, test_model, optimise_model
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
        'roberta-biomedical-clinical': "PlanTL-GOB-ES/bsc-bio-ehr-es",
        'roberta-bne': "PlanTL-GOB-ES/roberta-base-bne"
    }
    checkpoint = checkpoints[args.model_name]
    # beto
    if checkpoint == "dccuchile/bert-base-spanish-wwm-cased":
        hyperparameters = {
            'num_train_epochs': 58,
            'learning_rate': 3.4268553890214325e-05, 
            'seed': 326, 
            'per_device_train_batch_size': 32, 
            'per_device_eval_batch_size': 16, 
            'weight_decay': 2.8436289860950645e-08, 
            'adam_epsilon': 2.4799103776060603e-09, 
            'warmup_steps': 203
            }
    # biobert
    elif checkpoint == "mrojas/bio-bert-base-spanish-wwm-cased":
        hyperparameters = {
            'num_train_epochs': 68,
            'learning_rate': 3.349363847683222e-05, 
            'seed': 326, 
            'per_device_train_batch_size': 32, 
            'per_device_eval_batch_size': 32, 
            'weight_decay': 0.01214452830676255, 
            'adam_epsilon': 3.039596615397574e-08, 
            'warmup_steps': 491
            }
    # roberta-biomedical-clinical
    elif checkpoint == "PlanTL-GOB-ES/bsc-bio-ehr-es":
        hyperparameters = {
            'num_train_epochs': 39,
            'learning_rate': 4.540969453284462e-05, 
            'seed': 324, 
            'per_device_train_batch_size': 32, 
            'per_device_eval_batch_size': 32, 
            'weight_decay': 0.00598936569463419, 
            'adam_epsilon': 1.724439344881123e-07, 
            'warmup_steps': 300
            } 
    # roberta-bne
    elif checkpoint == "PlanTL-GOB-ES/roberta-base-bne":
        hyperparameters = {
            'num_train_epochs': 50,
            'learning_rate': 3.0290898801655698e-05, 
            'seed': 320, 
            'per_device_train_batch_size': 32, 
            'per_device_eval_batch_size': 16, 
            'weight_decay': 4.5126980713116176e-08, 
            'adam_epsilon': 6.447418463180699e-08, 
            'warmup_steps': 125
            }
    train_model(checkpoint, hyperparameters=hyperparameters)

elif args.do == 'optimise':
    checkpoints = {
        'bio-bert-spanish': "mrojas/bio-bert-base-spanish-wwm-cased",
        'bert-base-spanish': "dccuchile/bert-base-spanish-wwm-cased",
        'roberta-biomedical-clinical': "PlanTL-GOB-ES/roberta-base-biomedical-es",
        'roberta-bne': "PlanTL-GOB-ES/roberta-base-bne"
    }
    checkpoint = checkpoints[args.model_name]
    optimise_model(checkpoint)

elif args.do == 'test':
    checkpoints = {
        'bert-base-spanish': 'chizhikchi/cares-bert-base',
        'bio-bert-spanish': 'chizhikchi/cares-biobert-base',
        'roberta-biomedical-clinical': 'chizhikchi/cares-roberta-clinical',
        'roberta-bne': 'chizhikchi/cares-roberta-bne'
    }
    checkpoint = checkpoints[args.model_name]
     # beto
    if checkpoint == 'chizhikchi/cares-bert-base':
        hyperparameters = {
            'num_train_epochs': 58,
            'learning_rate': 3.4268553890214325e-05, 
            'seed': 326, 
            'per_device_train_batch_size': 32, 
            'per_device_eval_batch_size': 16, 
            'weight_decay': 2.8436289860950645e-08, 
            'adam_epsilon': 2.4799103776060603e-09, 
            'warmup_steps': 203
            }
    # biobert 
    elif checkpoint == 'chizhikchi/cares-biobert-base':
        hyperparameters = {
            'num_train_epochs': 68,
            'learning_rate': 3.349363847683222e-05, 
            'seed': 326, 
            'per_device_train_batch_size': 32, 
            'per_device_eval_batch_size': 32, 
            'weight_decay': 0.01214452830676255, 
            'adam_epsilon': 3.039596615397574e-08, 
            'warmup_steps': 491
            }
    # roberta clinical
    elif checkpoint == 'chizhikchi/cares-roberta-clinical':
        hyperparameters = {
            'num_train_epochs': 39,
            'learning_rate': 4.540969453284462e-05, 
            'seed': 324, 
            'per_device_train_batch_size': 32, 
            'per_device_eval_batch_size': 32, 
            'weight_decay': 0.00598936569463419, 
            'adam_epsilon': 1.724439344881123e-07, 
            'warmup_steps': 300
            }
    # roberta bne
    elif checkpoint == 'chizhikchi/cares-roberta-bne':
        hyperparameters = {
            'num_train_epochs': 50,
            'learning_rate': 3.0290898801655698e-05, 
            'seed': 320, 
            'per_device_train_batch_size': 32, 
            'per_device_eval_batch_size': 16, 
            'weight_decay': 4.5126980713116176e-08, 
            'adam_epsilon': 6.447418463180699e-08, 
            'warmup_steps': 125
            }
    test_model(checkpoint, hyperparameters=hyperparameters)