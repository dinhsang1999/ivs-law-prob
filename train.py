import argparse
import os
import torch
import json
import warnings
import pandas as pd
import numpy as np
from tqdm import tqdm

from transformers import  AutoTokenizer
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from src.utils import set_seed
from src.dataset import Dataset
# from src.trainer import train
from src.model import PhoBERT_MultiLabel
#####################------------------------------------##############################

def cross_validate():

    # Create a KFold instance
    set_fold = KFold(n_splits=args.n_folds, shuffle=True, random_state = 69)

    #Load csv
    df = pd.read_csv(args.path_csv)

    #Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    #Create the dataset
    label_columns = df.columns.tolist()[1:-3] #FIXME:

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Use KFold to split the data into train and test sets
    for k_fold, (train_index, val_index) in enumerate(set_fold.split(df)):
        train_dataset = Dataset(df,tokenizer=tokenizer,max_token_len=args.max_token_len,label_columns=label_columns)
        val_dataset = Dataset(df,tokenizer=tokenizer,max_token_len=args.max_token_len,label_columns=label_columns)

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_index)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_index)

        train_dataloader = DataLoader(train_dataset,batch_size=args.batch_size,num_workers=args.num_worker,sampler=train_subsampler)
        val_dataloader = DataLoader(val_dataset,batch_size=args.batch_size,num_workers=args.num_worker,sampler=val_subsampler)

        model = PhoBERT_MultiLabel(args.n_classes,args.model_path)
        model = model.to(device)
        # Define the loss function and optimizer
        optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)

        for epoch in range(args.n_epochs):
            model.train()
            for step, batch in enumerate(train_dataloader):
                b_input_ids  = batch["input_ids"]
                b_attention_masks = batch["attention_mask"]
                b_labels = batch["labels"]
                b_input_ids = b_input_ids.to(device)
                b_attention_masks = b_attention_masks.to(device)
                b_labels = b_labels.to(device)

                # Forward pass
                loss, outputs = model(b_input_ids, attention_mask=b_attention_masks,labels=b_labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
            # Evaluate the model
            model.eval()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            for batch in val_dataloader:
                b_input_ids  = batch["input_ids"]
                b_attention_masks = batch["attention_mask"]
                b_labels = batch["labels"]
                b_input_ids = b_input_ids.to(device)
                b_attention_masks = b_attention_masks.to(device)
                b_labels = b_labels.to(device)
                with torch.no_grad():
                        outputs = model(b_input_ids, attention_mask=b_attention_masks, labels=b_labels)
                        loss = outputs[0]
                        logits = outputs[1]
                eval_loss += loss.mean().item()
                eval_accuracy += torch.sum(torch.argmax(logits, dim=1) == b_labels).item()
                nb_eval_examples += b_input_ids.size(0)
                nb_eval_steps += 1
            
            eval_loss = eval_loss / nb_eval_steps
            eval_accuracy = eval_accuracy / nb_eval_examples

            print("Loss: ", eval_loss)
            print("Accuracy: ", eval_accuracy)

#################-----------------#######################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--trial', type=str, default="0")
    parser.add_argument('--label-type', type=str, default="multi-label")

    parser.add_argument('--path-csv', type=str, default="/home/dhsang/BERT/data_full.csv") #CSV file
    parser.add_argument('--model-path', type=str, default="vinai/phobert-base") 
    parser.add_argument('--tokenizer-path', type=str, default="vinai/phobert-base") 
    parser.add_argument('--model-save', type=str, default="/mnt/data_lab513/dhsang/") #Should HHD disk

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num-worker', type=int, default=4)
    parser.add_argument('--n-classes', type=int, default=13)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--n-epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max-token-len', type=int, default=128)

    parser.add_argument('--ignore-warnings', action='store_false')

    args, _ = parser.parse_known_args()

    #Save variables to .txt - trial name
    os.makedirs("./trial_info",exist_ok=True)
    save_info_path = os.path.join("./trial_info","trial_"+str(args.trial))
    with open(save_info_path+".txt", 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    print('trial:',args.trial)
    print('SEED:',args.seed)
    set_seed(args.seed)

    if args.ignore_warnings:
        print("Ignored  warning!!!")
        warnings.filterwarnings("ignore")
    
    cross_validate()
    

    



