import torch
from model import Transformer
from transformers import AutoTokenizer  # pip install transformers
from utils import (
    BATCH_SIZE,
    BLOCK_SIZE,
    DEVICE,
    DROPOUT,
    LEARNING_RATE,
    NUM_EMBED,
    NUM_HEAD,
    NUM_LAYER,
    MAX_ITER,
    EVAL_INTER,
    encode,
    decode,
    get_batch,
    save_model_to_chekpoint,
    estimate_loss,
)
from config import *
from utils import *

def runner():
    print("Starting....")

    data_raw = open(PATH, encoding="utf-8").read()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size

    data = encode(text_seq=data_raw, tokenizer=tokenizer)
    n = int(0.9 * len(data))  # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]

    model = Transformer(
        vocab_size=vocab_size,
        num_embed=NUM_EMBED,
        block_size=BLOCK_SIZE,
        num_heads=NUM_HEAD,
        num_layers=NUM_LAYER,
        dropout=DROPOUT,
    )

    m = model.to(DEVICE)

    print(
        "Model with {:.2f}M parameters".format(sum(p.numel() for p in m.parameters()) / 1e6)
    )

    optimizer = torch.optim.AdamW(m.parameters(), lr=LEARNING_RATE)

    training_loop(m, optimizer, MAX_ITER, train_data, val_data)

    print("Validation....")

    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    print(
        decode(
            enc_sec=m.generate(idx=context, max_new_tokens=100, block_size=BLOCK_SIZE)[0],
            tokenizer=tokenizer,
        )
    )

runner()