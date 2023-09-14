import os
import torch
from datetime import datetime

# hyperparameters
BATCH_SIZE = 32  # how many independent sequences will we process in parallel?
BLOCK_SIZE = 64  # what is the maximum context length for predictions?
MAX_ITER = 5000  # number of training iterations
EVAL_INTER = 500
LEARNING_RATE = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_HEAD = 6
NUM_EMBED = NUM_HEAD * 128
NUM_LAYER = 6
DROPOUT = 0.2


def encode(text_seq: str, tokenizer: any) -> torch.Tensor:
    """
    Function to encode input text using a pre-trained tokenizer and vectorized lookups
    """
    # tokenize the input text
    tokens = tokenizer.tokenize(text_seq)
    # convert the tokens to their corresponding ids
    token_indices = tokenizer.convert_tokens_to_ids(tokens)
    token_indices = torch.tensor(token_indices, dtype=torch.long)
    return token_indices


def decode(enc_sec: torch.Tensor, tokenizer: any) -> str:
    """
    Function to decode a sequence of token indices back to a string
    """
    # convert the indices to a list
    enc_sec = enc_sec.tolist()
    # decode the indices to a string
    text = tokenizer.decode(enc_sec)
    return text


def get_batch(data, block_size: int, batch_size: int):
    """
    This is a simple function to create batches of data.
    GPUs allow for parallel processing we can feed multiple chunks at once
    so that's why we would need batches - how many independant sequences
    will we process in parallel.

    Parameters:
    data: list[str]: data to take batch from
    block_size (int): size of the text that is proccessed at once
    batch_size (int): number of sequences to process in parallel

    Returns:
    x, y: a tuple with token sequence and token target
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # we stack batch_size rows of sentences
    # so x and y are the matrices with rows_num=batch_size
    # and col_num=block_size
    x = torch.stack([data[i : i + block_size] for i in ix])
    # y is x shifted one position right - because we predict
    # word in y having all the previous words as context
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y


@torch.no_grad()
def estimate_loss(
    data,
    model: torch.nn.Module,
    block_size: int,
    batch_size: int,
    eval_iters: int = 10,
):
    out = {}
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(data=data, block_size=block_size, batch_size=batch_size)
        logits, loss = model.forward(X, Y)
        losses[k] = loss.item()
    out = losses.mean()
    model.train()
    return out


def load_model_from_checkpoint(
    model_class: torch.nn.Module,
    path_to_checkpoint: str = "checkpoints/state_dict_model.pt",
    **kwargs: dict,
) -> torch.nn.Module:
    try:
        state_dict = torch.load(path_to_checkpoint)
        print("Successfully loaded model from the checkpoint")
    except Exception as e:
        print(f"Error loading the model from the checkpoint. {e}")

    model = model_class(**kwargs)
    # load the state_dict into the model
    model.load_state_dict(state_dict)
    return model


def save_model_to_chekpoint(
    model: torch.nn.Module, path_to_checkpoint: str = "checkpoints", epoch: int = 0
):
    # check if path exists, otherwise create it
    if not os.path.exists(path_to_checkpoint):
        os.makedirs(path_to_checkpoint)

    # datetime object containing current date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d.%m.%Y_%H:%M:%S")
    checkpoint_name = "checkpoint_epoch-" + str(epoch) + "_" + dt_string + ".pt"
    full_path = os.path.join(path_to_checkpoint, checkpoint_name)
    try:
        torch.save(model.state_dict(), full_path)
        print("Successfully saved the model to {}".format(full_path))
    except Exception as e:
        print(f"Error saving the model to checkpoint. {e}")

def training_loop(m, optimizer, max_iter, train_data, val_data):
    for step in range(max_iter):

        # every EVAL_INTER evaluate the loss on train and val sets
        if step % EVAL_INTER == 0 or step == max_iter - 1:
            loss_train = estimate_loss(
                data=train_data, model=m, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE
            )
            loss_val = estimate_loss(
                data=val_data, model=m, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE
            )
            print("step {:10} | train loss {:6.4f} | val loss {:6.4f}".format(step, loss_train, loss_val))

        # sample a batch of data
        xb, yb = get_batch(data=train_data, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE)
        logits, loss = m.forward(xb, yb)
        # zero_grad() method sets the gradients of all parameters in the optimizer to zero
        optimizer.zero_grad(set_to_none=True)
        # backward() method on the loss variable calculates the gradients 
        # of the loss with respect to the model's parameters.
        loss.backward()
        # step() method on the optimizer updates the model's parameters 
        # using the calculated gradients, in order to minimize the loss.
        optimizer.step()