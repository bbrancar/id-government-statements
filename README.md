# Identifying Government Statements
---
This repo contains a Colab notebook and accompanying `.py` and `.json` files containing a PyTorch implementation of the  `hfl/chinese-roberta-wwm-ext` transformer for Chinese text classification. The model was employed to identify Chinese news articles directly covering government meetings and official statements. Using a training set of 500 articles, a validation set of 200 articles, and a test set of 200 articles, the transformer achieved an F1-score of over 94%. 

# Repurposing Code
---
In order to employ this code with new datasets, changes must be made to the `.ipynb` notebook. In particular, the `ROOT` variable must be set to folder containing `.py` files, and the data variable names must be altered in the final cells. Changes must also be made to the `load_data.py` file, including: changes in variable names within the  ` __getitem__`, `get_dataset`, and `inference_data_processing` functions. Finally, the data paths must be updated within the `roberta_config.json` file.

In order to explore the use of alternative transformer models, replace the `"transformer_name"` variable the relevant transformer from the HuggingFace model library (https://huggingface.co/models). The current algorithm will handle variations of `BERT` and `RoBERTa`, however likely requires minor tweaks in the case of other transformers.
