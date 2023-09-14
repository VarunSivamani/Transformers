import torch.optim as optim
import torch.nn as nn
import numpy as np

# =============================================================================
# Methods / Class
# =============================================================================
def get_batch(loader, loader_iter):
    try:
        batch = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        batch = next(loader_iter)
    return batch, loader_iter

def training_loop(model,loss_model, optimizer,data_loader, batch_iter, print_each, n_iteration):
    for it in range(n_iteration):
        
        #get batch
        batch, batch_iter = get_batch(data_loader, batch_iter)
        
        #infer
        masked_input = batch['input']
        masked_target = batch['target']
        
        masked_input = masked_input.cuda(non_blocking=True)
        masked_target = masked_target.cuda(non_blocking=True)
        output = model(masked_input)
        
        #compute the cross entropy loss 
        output_v = output.view(-1,output.shape[-1])
        target_v = masked_target.view(-1,1).squeeze()
        loss = loss_model(output_v, target_v)
        
        #compute gradients
        loss.backward()
        
        #apply gradients
        optimizer.step()
        
        #print step
        if it % print_each == 0:
            print('it:', it, 
                ' | loss', np.round(loss.item(),2),
                ' | Δw:', round(model.embeddings.weight.grad.abs().sum().item(),3))
        
        #reset gradients
        optimizer.zero_grad()