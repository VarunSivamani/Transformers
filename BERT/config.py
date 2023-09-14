import os

BATCH_SIZE = 1024
SEQ_LEN = 20
EMBED_SIZE = 128
INNER_FF_SIZE = EMBED_SIZE * 4
N_HEADS = 8
N_CODE = 8
N_VOCAB = 40000
DROPOUT = 0.1

OPTIM_KWARGS = {'lr':1e-4, 'weight_decay':1e-4, 'betas':(.9,.999)}

PTH = './data/'