# CARES - A Corpus of Anonymised Radiological Evidences in Spanish

This repository contains code to reproduce the experiments described in the paper **CARES: A Corpus for Classification of Spanish Radiological Reports**. 
CARES is a high-quality text resource manually labeled with ICD-10 codes and reviewed by radiologists. These types of resources are essential for developing automatic text classification tools as they are necessary for training and fine-tuning our computational systems. The dataset is available on [HuggingFace hub](//huggingface.co/datasets/chizhikchi/CARES)

## Corpus description and statistics 
The CARES corpus has been manually annotated using the ICD-10 ontology, which stands for for the 10th version of the International Classification of Diseases. For each radiological report, a **minimum of one code** and a **maximum of 9 codes** were assigned, while the average number of codes per text is 2.15
with the standard deviation of 1.12. 

The corpus was additionally preprocessed in order to make its format coherent with the automatic text classification task. Considering the hierarchical structure of the ICD-10 ontology, each sub-code was mapped to its respective code and chapter, obtaining two new sets of labels for each report. The entire CARES collection contains 6,907 sub-code annotations among the
3,219 radiologic reports. There are 223 unique ICD-10 sub-codes within the annotations, which were mapped to 156 unique ICD-10 codes and 16 unique chapters of the cited ontology.

## Text classification experiments 

The main objective of this repository is to favor the repropuctibility of the experiments descibed in the paper. These experementation focused at developing system to classify ach report with corresponding ICD-10 chapters. 
For this purpose, we performed a stratified split of the corpus into train (70\%) and test (30\%) subsets where the former was used to fine-tune the models and the latter to evaluate them. 

The experimentation process consisted in fine-tuning three pre-trained transformer models:

* [BETO](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased) - a general domain Spanish BERT model 
* [RoBERTa-bne](https://huggingface.co/PlanTL-GOB-ES/roberta-base-bne) - a Spanish RoBERTa model considered SOTA in general domain Spanish NLP
* [BioBERT-Spanish](https://github.com/plncmm/bio-bert-base-spanish-wwm-uncased.git) - BERT general model extended with domain-specific knowledge by fine-tuning over a Chilean clinical corpus.
* [RoBERTa-biomedical-clinical](https://huggingface.co/PlanTL-GOB-ES/roberta-base-biomedical-clinical-es) - a RoBERTa-based model pretrained on a combination of biomedical and clinical corpora.

The table below summarises the results obtained by each developed system. 

<table>
  <tr>
    <td></td>
    <td colspan=3>Micro-avg</td>
    <td colspan=3>Macro-avg</td>
  </tr>
  <tr>
    <td>Model</td>
    <td>Precision</td>
    <td>Recall</td>
    <td>F1-score</td>
    <td>Precision</td>
    <td>Recall</td>
    <td>F1-score</td>
  </tr>
  <tr>
    <td> BETO </td>
    <td> 0.8687 </td>
    <td> 0.8464 </td>
    <td> 0.8574 </td>
    <td> 0.8673 </td>
    <td> 0.7975 </td>
    <td> 0.8250 </td>
  </tr>
    <tr>
    <td> RoBERTa-BNE </td>
    <td> 0.9032 </td>
    <td> 0.7937 </td>
    <td> 0.8449 </td>
    <td> 0.9303 </td>
    <td> 0.6845 </td>
    <td> 0.7673 </td>
  </tr>
  <tr>
    <td> BioBERT </td>
    <td> 0.7813 </td>
    <td> 0.7396 </td>
    <td> 0.7599 </td>
    <td> 0.8296 </td>
    <td> 0.6817 </td>
    <td> 0.7365 </td>
  </tr>
  <tr>
    <td> RoBERTa Biomedical </td>
    <td> 0.8562 </td>
    <td> 0.8794 </td>
    <td><b> 0.8676 </b></td>
    <td> 0.8740 </td>
    <td> 0.8213 </td>
    <td><b> 0.8328 </b></td>
  </tr>
</table>

You can reproduce the whole fine-tuning process with hyperparameter optimisation by running: 

```
cd CARES
pip install requirements.txt
python main.py --model NAME OF THE MODEL --do optimise
```
train models with the hyperparameters we selected in our experimentation by running:

```
cd CARES
pip install requirements.txt
python main.py --model NAME OF THE MODEL --do train
```

or just run testing on the fine-tuned models available on HuggingFace by running:

```
cd CARES
pip install requirements.txt
python main.py --model NAME OF THE MODEL --do test
```

Note that you must select the model to train or test among the following options: **bio-bert-spanish**, **roberta-biomedical-clinical**, **bert-base-spanish**, **roberta-bne**.
