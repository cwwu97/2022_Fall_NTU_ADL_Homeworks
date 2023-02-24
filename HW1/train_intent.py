import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import trange, tqdm

from dataset import SeqClsDataset
from utils import Vocab
from model import SeqClassifier

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(args):
    print('CUDA:', torch.cuda.is_available())
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    # Load intent to index mapping dict
    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    # Load train data
    train_data_path = Path(args.data_dir / f"{TRAIN}.json")
    train_data = json.loads(train_data_path.read_text())
    train_dataset = SeqClsDataset(train_data, vocab, label_mapping=intent2idx, max_len=args.max_len, test_mode=False)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)

    # Load eval data
    eval_data_path = Path(args.data_dir / f"{DEV}.json")
    eval_data = json.loads(eval_data_path.read_text())
    eval_dataset = SeqClsDataset(eval_data, vocab, label_mapping=intent2idx, max_len=args.max_len, test_mode=False)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=eval_dataset.collate_fn)

    # Load word embedding
    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqClassifier(
        embeddings=embeddings,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        num_class=len(intent2idx))
    model.to(DEVICE)

    # TODO: init optimizer
    for param in model.parameters():
        param.requires_grad = True
    optimizer = AdamW(model.parameters(), lr = args.lr)
    cross_entropy = CrossEntropyLoss()

    num_epoch = args.num_epoch
    best_acc = 0
    for epoch in range(num_epoch):

        # TODO: Training loop - iterate over train dataloader and update model weights
        print(f'Epoch {epoch+1}/{num_epoch} | Training and Evaluating...')
        model.train()
        sum_train_loss, avg_train_acc = 0, 0
        for idx, (texts, intents) in enumerate(tqdm(train_dataloader)):
            # get x and y
            texts = texts.to(DEVICE, dtype=torch.long)
            intents = intents.to(DEVICE, dtype=torch.long)
            # reset gradient
            optimizer.zero_grad()
            # model predict
            preds = model(texts)
            # calculate cross entropy loss
            loss = cross_entropy(preds, intents)
            # calculate gradient
            loss.backward()
            # update weights by back propagate
            optimizer.step()
            # evaluate
            sum_train_loss += (loss.item()/len(train_dataloader))

        # TODO: Evaluation loop - calculate accuracy and save model weights
        model.eval()
        total_preds = []
        total_intents = []
        sum_eval_loss, avg_eval_acc = 0, 0
        with torch.no_grad():
            for idx, (texts, intents) in enumerate(tqdm(eval_dataloader)):
                # get x and y
                texts = texts.to(DEVICE, dtype=torch.long)
                intents = intents.to(DEVICE, dtype=torch.long)     
                # model predict
                preds = model(texts)
                # calculate cross entropy loss
                loss = cross_entropy(preds, intents)
                # aggregate evaluation prediction
                preds = torch.argmax(preds, dim=1)
                total_preds.extend(preds.detach().cpu().numpy())
                total_intents.extend(intents.detach().cpu().numpy())
                # evaluate
                sum_eval_loss += (loss.item()/len(eval_dataloader))

        eval_acc = (np.array(total_preds) == np.array(total_intents)).sum()/len(eval_data)
        if eval_acc > best_acc:
            best_acc = eval_acc
            torch.save(model.state_dict(), Path(args.ckpt_dir / 'intent_best.pt'))

        print(f'Training loss: {sum_train_loss:.5f}')
        print(f'Evaluating loss: {sum_eval_loss:.5f} | acc: {eval_acc:.5f}')
        print('\n')

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./",
    )

    # data
    parser.add_argument("--max_len", type=int, default=32)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=40)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
