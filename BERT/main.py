from collections import Counter
from os.path import exists
import re
import torch
import torch.optim as optim
import torch.nn as nn
from config import *
from utils import *
from dataset import SentencesDataset
from model import Transformer

def runner():
    print('loading text...')
    sentences = open(PTH+'training.txt').read().lower().split('\n')
    
    print('tokenizing sentences...')
    special_chars = ',?;.:/*!+-()[]{}"\'&'
    sentences = [re.sub(f'[{re.escape(special_chars)}]', ' \g<0> ', s).split(' ') for s in sentences]
    sentences = [[w for w in s if len(w)] for s in sentences]

    print('creating/loading vocab...')
    pth = PTH + 'vocab.txt'
    if not exists(pth):
        words = [w for s in sentences for w in s]
        vocab = Counter(words).most_common(N_VOCAB) #keep the N most frequent words
        vocab = [w[0] for w in vocab]
        open(pth, 'w+').write('\n'.join(vocab))
    else:
        vocab = open(pth).read().split('\n')

    print('creating dataset...')
    dataset = SentencesDataset(sentences, vocab, SEQ_LEN)
    # kwargs = {'num_workers':n_workers, 'shuffle':True,  'drop_last':True, 'pin_memory':True, 'batch_size':batch_size}
    kwargs = {'shuffle':True,  'drop_last':True, 'pin_memory':True, 'batch_size':BATCH_SIZE}
    data_loader = torch.utils.data.DataLoader(dataset, **kwargs)

    print('initializing model...')
    model = Transformer(N_CODE, N_HEADS, EMBED_SIZE, INNER_FF_SIZE, len(dataset.vocab), SEQ_LEN, DROPOUT)
    model = model.cuda()

    print('initializing optimizer and loss...')
    optimizer = optim.Adam(model.parameters(), **OPTIM_KWARGS)
    loss_model = nn.CrossEntropyLoss(ignore_index=dataset.IGNORE_IDX)

    print('training...')
    print_each = 10
    model.train()
    batch_iter = iter(data_loader)
    n_iteration = 10000

    training_loop(model,loss_model,optimizer,data_loader,batch_iter,print_each,n_iteration)

    print('saving embeddings...')
    N = 3000
    np.savetxt('values.tsv', np.round(model.embeddings.weight.detach().cpu().numpy()[0:N], 2), delimiter='\t', fmt='%1.2f')
    s = [dataset.rvocab[i] for i in range(N)]
    open('names.tsv', 'w+').write('\n'.join(s) )

    print("Succees")
    print('end')

runner()