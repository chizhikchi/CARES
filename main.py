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
        lr = 5e-5
        bs = 16
    elif checkpoint == "dccuchile/bert-base-spanish-wwm-cased":
        lr = 3e-5
        bs = 16
    elif checkpoint == "PlanTL-GOB-ES/roberta-base-biomedical-es":
        lr = 5e-5 
        bs = 32
    train_model(checkpoint, learning_rate=lr, batch_size=bs)
        
elif args.do == 'test':
    checkpoints = {
        'bio-bert-spanish': 'chizhikchi/cares-biobert-base',
        'bert-base-spanish': 'chizhikchi/cares-bert-base',
        'roberta-biomedical-clinical': 'chizhikchi/cares-roberta-clinical'
    }
    checkpoint = checkpoints[args.model_name]
    if checkpoint == 'chizhikchi/cares-biobert-base':
        lr = 5e-5
        bs = 16
    elif checkpoint == 'chizhikchi/cares-bert-base':
        lr = 3e-5
        bs = 16
    elif checkpoint == 'chizhikchi/cares-roberta-clinical':
        lr = 5e-5 
        bs = 32
    test_model(checkpoint, learning_rate=lr, batch_size=bs)