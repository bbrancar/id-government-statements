import torch
import os
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from tqdm import tqdm
from ast import literal_eval
import numpy as np
from transformers import AutoTokenizer

class ArticleDataset(Dataset):

    
    def __init__(self, dataframe, tokenizer, max_len, get_wids, inference):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.inference = inference

    def __getitem__(self, index):
        # GET TEXT AND WORD LABELS 
        text = self.data.text[index]

        if self.inference == False:    
          label = self.data.label[index]  
        
        article_id = self.data.cnki_id[index]

        # TOKENIZE TEXT
        encoding = self.tokenizer(text, 
                                padding='max_length', 
                                truncation=True, 
                                max_length=self.max_len)
 
        # CONVERT TO TORCH TENSORS
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        if self.inference == False:
          item['label'] = torch.as_tensor(label)
        
        item['cnki_id'] = article_id
        return item

    def __len__(self):
        return self.len

def get_dataset(config_data, tokenizer, labels, outputs, lab_col):
    
    #Pull in dataset
    df = pd.read_csv(config_data['dataset']['data_path'])
    df['label'] = df[lab_col].map(dict(zip(labels, outputs)))

    #Keep original
    df['idx'] = df.index.values

    print('There are',len(df),'articles. We will split 70%-15%-15% for validation and testing.')

    # TRAIN VALID TEST SPLIT
    np.random.seed(23)
    permuted_idx = np.random.permutation(len(df))
    train_idx = permuted_idx[:int(0.7 * len(df))]
    val_idx = permuted_idx[int(0.7 * len(df)): int(0.85 * len(df))]
    test_idx = permuted_idx[int(0.85 * len(df)):]
    np.random.seed(None)

    # CREATE TRAIN SUBSET AND VALID SUBSET
    data = df[['cnki_id', 'text', 'label']]
    train_dataset = data.loc[train_idx].reset_index(drop=True)
    val_dataset = data.loc[val_idx].reset_index(drop=True)
    test_dataset = data.loc[test_idx].reset_index(drop=True)

    # tokenizer = AutoTokenizer.from_pretrained(config_data['model']['transformer_path']) 
    training_set = ArticleDataset(train_dataset, tokenizer, config_data['experiment']['max_length'], False, False)
    validation_set = ArticleDataset(val_dataset, tokenizer, config_data['experiment']['max_length'], True, False)
    testing_set = ArticleDataset(test_dataset, tokenizer, config_data['experiment']['max_length'], True, False)

    train_loader = DataLoader(training_set,
                              batch_size=config_data['experiment']['batch_size'],
                              shuffle=True, num_workers=config_data['experiment']['num_workers'],
                              pin_memory=True)
    val_loader = DataLoader(validation_set,
                              batch_size=config_data['experiment']['batch_size'],
                              shuffle=False, num_workers=config_data['experiment']['num_workers'],
                              pin_memory=True)
    test_loader = DataLoader(testing_set,
                              batch_size=config_data['experiment']['batch_size'],
                              shuffle=False, num_workers=config_data['experiment']['num_workers'],
                              pin_memory=True)
    
    train_df = df.loc[train_idx].reset_index(drop=True)
    val_df = df.loc[val_idx].reset_index(drop=True)
    test_df = df.loc[test_idx].reset_index(drop=True)
                              
    return train_loader, val_loader, test_loader, val_df, test_df, train_df

def inference_data_processing(config_data, tokenizer):

    #Pull in dataset
  df = pd.read_csv(config_data['dataset']['infer_data_path'])

  #Keep original
  df['idx'] = df.index.values

  # CREATE TRAIN SUBSET AND VALID SUBSET
  infer_dataset = df[['cnki_id', 'text']]


  # tokenizer = AutoTokenizer.from_pretrained(config_data['model']['transformer_path']) 
  infer_set = ArticleDataset(infer_dataset, tokenizer, config_data['experiment']['max_length'], False, True)

  infer_loader = DataLoader(infer_set,
                            batch_size=config_data['experiment']['batch_size'],
                            shuffle=False, num_workers=config_data['experiment']['num_workers'],
                            pin_memory=True)

  return infer_loader, infer_dataset
