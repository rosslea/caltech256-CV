import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from tqdm import tqdm
import os
import pickle
import sys
import argparse
import json
from typing import Tuple, Optional, Union
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class ClfDataset(Dataset):
    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        prefix = self.prefixes[item]
        lable = self.labels[item]
        return prefix, lable

    def __init__(self, data_path: str):

        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)

        self.prefixes = all_data["clip_embedding"]
        labels = all_data["labels"]
        labels = [i[:3] for i in labels]
        for l in labels:
            if l[0][0] == '0':
                if l[0][1] == '0':
                    l[0] = l[0][2]
                else:
                    l[0] = l[0][1:3]
            else:
                l[0] = l[0][:3]
        labels = [int(i[0])-1 for i in labels]
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

class MLP(nn.Module):

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super().__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x.type(torch.float32))
        return out

class ClfModel(nn.Module):

    def forward(self, prefix: torch.Tensor):

        prefix_projections = self.clip_project(prefix)
        out = self.softmax(prefix_projections)
        return out

    def __init__(self, prefix_length: int = 512):
        super().__init__()
        
        self.clip_project = MLP((prefix_length, prefix_length, prefix_length, 257))
        self.softmax = nn.LogSoftmax(dim=1)
      


def train(dataset, validset, model: ClfModel, args,
          lr: float = 2e-5, warmup_steps: int = 5000, output_dir: str = ".", output_prefix: str = ""):

    ## [cfg]
    device = torch.device(f'cuda:{args.cuda_num}')
    # device = torch.device('cpu')
    batch_size = args.bs
    epochs = args.epochs
    n_workers = args.num_workers
    if not Path(output_dir).exists():
        Path(output_dir).mkdir()
    ## [cfg]
    
    ## [model data opotim loss]
    model = model.to(device)
    model.train()
    
    train_dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=n_workers)
    # train_dataloader = DataLoader(dataset, batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters())
    
    loss_func = nn.NLLLoss()
    ## [model data optim loss]

    # save_config(args)

    ## [loss acc]
    loss_train, loss_valid, acc_train, acc_valid = [], [], [], []
    ## [loss acc]

    ## [train]
    for epoch in range(epochs):
        print(f">>> Training epoch {epoch}")
        train_loss, train_acc, counter_train = 0, 0, 0 


        ## [iter on dataloader]
        progress = tqdm(total=len(train_dataloader), desc=output_prefix)
        for idx, (prefix, labels) in enumerate(train_dataloader):

            ## [forward]
            model.zero_grad()
            prefix = prefix.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.int64)
            outputs = model(prefix)
            ## [forward]

            loss = loss_func(outputs, labels)
            train_loss += loss.item() * labels.size(0)
            _values, predictions = torch.max(outputs.data, 1)
            correct_prediction_counts = predictions.eq(labels)

            # Convert correct_prediction_counts to float and then compute the mean
            acc = torch.mean(correct_prediction_counts.type(torch.FloatTensor))

            # Compute total accuracy in the whole batch and add to valid_acc
            train_acc +=acc.item() * len(labels)
            counter_train += len(labels)

            
            ## [backward]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            ## [backward]

            ## [metrics]
            progress.set_postfix({"loss": loss.item()}) # print a metric
            progress.update() # progress bar
            ## [metrics]

        progress.close()
        ## [iter on dataloader]


        ## [checkpoint]
        if epoch % args.save_every == 0 or epoch == epochs - 1:
            torch.save(
                model.state_dict(),
                Path(output_dir)/Path(f"{output_prefix}-{epoch:03d}.pt"),
            )
        ## [checkpoint]
    
        ## [performance per epoch]
        with torch.no_grad():
            model.eval()
            validation_data = DataLoader(validset, batch_size, shuffle=True, num_workers=n_workers)
            valid_loss, valid_acc, counter_valid = 0, 0, 0

            ## [iter on dataloader]
            for j, (inputs, labels) in enumerate(validation_data):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = loss_func(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)
                _values, predictions = torch.max(outputs.data, 1)
                correct_prediction_counts = predictions.eq(labels)

                # Convert correct_prediction_counts to float and then compute the mean
                acc = torch.mean(correct_prediction_counts.type(torch.FloatTensor))

                # Compute total accuracy in the whole batch and add to valid_acc
                valid_acc +=acc.item() * len(labels)
                counter_valid += len(labels)
            ## [iter on dataloader]

        ## [performance per epoch]

        ## [all performance]
        loss_valid.append(valid_loss)
        acc_valid.append(valid_acc/counter_valid)
        loss_train.append(train_loss)
        acc_train.append(train_acc/counter_train)
        with open('data/performance.pkl', 'wb') as f:
            pickle.dump(dict(loss_train=loss_train, loss_valid=loss_valid, acc_train=acc_train, acc_valid=acc_valid), f)
        print(f'performance saved')
        ## [all performance]




    ## [train]

    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data/ViT-B_32_train.pkl')
    parser.add_argument('--out_dir', default='./checkpoints')
    parser.add_argument('--prefix', default='train', help='prefix for saved filenames')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--prefix_length', type=int, default=10)
    parser.add_argument('--prefix_length_clip', type=int, default=10)
    parser.add_argument('--bs', type=int, default=10000)
    parser.add_argument('--cuda_num', type=int, default=2, help='0 to 3')
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    dataset_train = ClfDataset(args.data)
    dataset_valid = ClfDataset(args.data.replace('train', 'validation'))
    model = ClfModel(prefix_length=512)
    train(dataset_train, dataset_valid, model, args, output_dir=args.out_dir, output_prefix=args.prefix)


if __name__ == '__main__':
    main()