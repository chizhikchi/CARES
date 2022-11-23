# CARES - A Corpus of Anonymised Radiological Evidences in Spanish

This repository contain data, code and the fine-tuned models for the paper in which we present **CARES - a Corpus of Anonymised Radiological Evidences in Spanish**. 
CARES is a high-quality text resource manually labeled with ICD-10 codes and reviewed by radiologists. These types of resources are essential for developing automatic text classification tools as they are necessary for training and fine-tuning our computational systems.

## Corpus description and statistics 
The CARES corpus has been manually annotated using the ICD-10 ontology, which stands for for the 10th version of the International Classification of Diseases. For each radiological report, a **minimum of one code** and a **maximum of 9 codes** were assigned, while the average number of codes per text is 2.15
with the standard deviation of 1.12. 

The corpus was additionally preprocessed in order to make its format coherent with the automatic text classification task. Considering the hierarchical structure of the ICD-10 ontology, each sub-code was mapped to its respective code and chapter, obtaining two new sets of labels for each report. The entire CARES collection contains 6,907 sub-code annotations among the
3,219 radiologic reports. There are 223 unique ICD-10 sub-codes within the annotations, which were mapped to 156 unique ICD-10 codes and 16 unique chapters of the cited ontology.

## Text classification experiments 

The main objective of this repository is to favor the repropuctibility of the experiments descibed in the paper. These experementation focused at developing system to classify ach report with corresponding ICD-10 chapters. 
For this purpose, we performed a stratified split of the corpus into train (70\%) and test (30\%) subsets where the former was used to fine-tune the models and the latter to evaluate them. 
This data splits, alongside with the entire corpus, can be found in the [data](https://github.com/chizhikchi/CARES/tree/main/data) subfolder.

The experimentation process consisted in fine-tuning three pre-trained transformer models:

* [BETO](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased) - a general domain Spanish BERT model 
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
    <td> 0.8448 </td>
    <td> 0.7654 </td>
    <td> 0.7958 </td>
    <td> 0.7604 </td>
    <td> 0.6030 </td>
    <td> 0.6553 </td>
  </tr>
  <tr>
    <td> BioBERT </td>
    <td> 0.7404 </td>
    <td> 0.6785 </td>
    <td> 0.5742 </td>
    <td> 0.7327 </td>
    <td> 0.6920 </td>
    <td> 0.7081 </td>
  </tr>
  <tr>
    <td> RoBERTa Biomedical </td>
    <td> 0.8448 </td>
    <td> 0.7964 </td>
    <td><b> 0.8199 </b></td>
    <td> 0.8305 </td>
    <td> 0.6920 </td>
    <td><b> 0.7344 </b></td>
  </tr>
</table>

You can either reproduce the whole fine-tuning process by running 

```
cd CARES
pip install requirements.txt
python main.py --model NAME OF THE MODEL --do train
```

or just run testing on the fine-tuned models included in the [checkpoints](https://github.com/chizhikchi/CARES/tree/main/checkpoints) subfolder by running

```
cd CARES
pip install requirements.txt
python main.py --model NAME OF THE MODEL --do test
```

Note that you must select the model to train or test among the following options: **bio-bert-spanish**, **roberta-biomedical-clinical**, **bert-base-spanish**.
