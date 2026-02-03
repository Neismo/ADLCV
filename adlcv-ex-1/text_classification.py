import numpy as np
import os
import random
import torch
from torch import nn
import torch.nn.functional as F
from torchtext import data, datasets, vocab
import tqdm
import itertools

from transformer import TransformerClassifier, to_device

NUM_CLS = 2
VOCAB_SIZE = 50_000
SAMPLED_RATIO = 0.2
MAX_SEQ_LEN = 512

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def prepare_data_iter(sampled_ratio=0.2, batch_size=16):
    TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
    LABEL = data.Field(sequential=False)
    tdata, testdata = datasets.IMDB.splits(TEXT, LABEL)
    # Reduce dataset size
    reduced_tdata, _ = tdata.split(split_ratio=sampled_ratio)
    # Create train and test splits
    train, val = reduced_tdata.split(split_ratio=0.8)
    print('training: ', len(train), 'validation: ', len(val), 'test: ', len(testdata))
    TEXT.build_vocab(train, max_size= VOCAB_SIZE - 2)
    LABEL.build_vocab(train)
    train_iter, val_iter, test_iter = data.BucketIterator.splits((train, val, testdata), 
                                                       batch_size=batch_size, 
                                                       device=to_device()
    )

    return train_iter, val_iter, test_iter


def main(embed_dim=128, num_heads=4, num_layers=4, num_epochs=20,
         pos_enc='fixed', pool='max', dropout=0.0, fc_dim=None,
         batch_size=16, lr=1e-4, warmup_steps=625, 
         weight_decay=1e-4, gradient_clipping=1
    ):

    
    
    loss_function = nn.CrossEntropyLoss()

    train_iter, val_iter, test_iter = prepare_data_iter(sampled_ratio=SAMPLED_RATIO, 
                                            batch_size=batch_size
    )


    model = TransformerClassifier(embed_dim=embed_dim, 
                                  num_heads=num_heads, 
                                  num_layers=num_layers,
                                  pos_enc=pos_enc,
                                  pool=pool,  
                                  dropout=dropout,
                                  fc_dim=fc_dim,
                                  max_seq_len=MAX_SEQ_LEN, 
                                  num_tokens=VOCAB_SIZE, 
                                  num_classes=NUM_CLS,
                                  )
    
    if torch.cuda.is_available():
        model = model.to('cuda')

    opt = torch.optim.AdamW(lr=lr, params=model.parameters(), weight_decay=weight_decay)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / warmup_steps, 1.0))

    # training loop
    for e in range(num_epochs):
        print(f'Epoch {e}')
        model.train()
        for batch in tqdm.tqdm(train_iter):
            opt.zero_grad()
            input_seq = batch.text[0]
            batch_size, seq_len = input_seq.size()
            label = batch.label - 1
            if seq_len > MAX_SEQ_LEN:
                input_seq = input_seq[:, :MAX_SEQ_LEN]
            out = model(input_seq)
            loss = loss_function(out, label) # compute loss
            loss.backward() # backward
            # if the total gradient vector has a length > 1, we clip it back down to 1.
            if gradient_clipping > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            opt.step()
            sch.step()

        with torch.no_grad():
            model.eval()
            tot, cor= 0.0, 0.0
            for batch in val_iter:
                input_seq = batch.text[0]
                batch_size, seq_len = input_seq.size()
                label = batch.label - 1
                if seq_len > MAX_SEQ_LEN:
                    input_seq = input_seq[:, :MAX_SEQ_LEN]
                out = model(input_seq).argmax(dim=1)
                tot += float(input_seq.size(0))
                cor += float((label == out).sum().item())
            acc = cor / tot
            print(f'-- {"validation"} accuracy {acc:.3}')

    # final test
    with torch.no_grad():
        model.eval()
        tot, cor= 0.0, 0.0
        for batch in test_iter:
            input_seq = batch.text[0]
            batch_size, seq_len = input_seq.size()
            label = batch.label - 1
            if seq_len > MAX_SEQ_LEN:
                input_seq = input_seq[:, :MAX_SEQ_LEN]
            out = model(input_seq).argmax(dim=1)
            tot += float(input_seq.size(0))
            cor += float((label == out).sum().item())
        acc = cor / tot
        print(f'-- {"test"} accuracy {acc:.3}')
    
    return acc


def grid_sweep():
    sweep_config = {
        "embed_dim": [64, 128],
        "num_heads": [4, 8],
        "num_layers": [2, 4],
        "dropout": [0.1, 0.2],
        "lr": [1e-4, 5e-5]
    }

    keys, values = zip(*sweep_config.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    results = []
    print(f"Starting Grid Sweep: {len(combinations)} total combinations.")

    for i, config in enumerate(combinations):
        print(f"\n--- Run {i+1}/{len(combinations)} ---")
        print(f"Config: {config}")
        
        # Reset seeds for each run to ensure comparability
        set_seed(1)
        
        try:
            test_acc = main(**config)
            config['test_acc'] = test_acc
            results.append(config)
        except Exception as e:
            print(f"Run failed with error: {e}")

    # Sort results by accuracy descending
    results.sort(key=lambda x: x['test_acc'], reverse=True)

    # Save to file
    with open("sweep_results.txt", "w") as f:
        f.write("Grid Sweep Results (Sorted by Accuracy)\n")
        f.write("="*40 + "\n")
        for res in results:
            line = f"Acc: {res['test_acc']:.4f} | " + " | ".join([f"{k}: {v}" for k, v in res.items() if k != 'test_acc'])
            f.write(line + "\n")
            print(line)

    print("\nSweep complete. Results saved to sweep_results.txt")


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"]= str(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    print(f"Model will run on {device}")
    grid_sweep()
