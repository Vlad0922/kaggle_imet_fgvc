import numpy as np

from utils import sigmoid

def train_epoch(train_loader, model, loss_fn, optimizer, scheduler=None, tqdm_module=tqdm.tqdm, disable_tqdm=False,
               accum_steps = 1, mean_accum=False, scale_loss_bs=True, loss_reduction=None):
    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (data, target) in tqdm_module(enumerate(train_loader), total=len(train_loader), leave=False, disable=disable_tqdm):
        if not (type(data) in [list, tuple]):
            data = (data,)

        data = (d.cuda() for d in data)
        target = target.type(torch.float).cuda()

        outputs = model(*data)

        loss = loss_fn(outputs, target)
        
        if loss_reduction:
            loss = loss_reduction(loss)
        
        losses.append(loss.item())
        total_loss += loss.item()
        
        if mean_accum:
            loss /= accum_steps
        
        if scale_loss_bs:
            (loss * train_loader.batch_size).backward()   
        else:    
            loss.backward()
        
        if (batch_idx + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        if scheduler:
            scheduler.batch_step()
    
    if (batch_idx + 1) % accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    total_loss /= (batch_idx + 1)
    return np.mean(losses)


def test_epoch(val_loader, model, loss_fn, loss_reduction=None):
    with torch.no_grad():
        model.eval()

        losses = list()
        for batch_idx, (data, target) in enumerate(val_loader):
            if not (type(data) in [list, tuple]):
                data = (data,)

            data = (d.cuda() for d in data)
            target = target.type(torch.float).cuda()

            outputs = model(*data)
            loss = loss_fn(outputs, target)
            
            if loss_reduction:
                loss = loss_reduction(loss)

            losses.append(loss.item())

    return np.mean(losses)

def predict(model, loader, tqdm_module=tqdm_notebook, disable_tqdm=True):
    res = list()

    with torch.no_grad():
        model.eval()

        for (data, target) in tqdm_module(loader, disable=disable_tqdm):
            if not (type(data) in [list, tuple]):
                data = (data,)

            data = (d.cuda() for d in data)
            outputs = model(*data).cpu().numpy()
            res.append(outputs)

    return np.vstack(np.array(res))

def train_cycle(model, loss_fn, optimizer, scheduler, train_loader, val_loader, val_labels, bs, loss_reduction=reduce_loss, threshold=0.1, iterate_threshold=False):
    accum_steps = bs // train_loader.batch_size
    sigmoid_out = model.sigmoid
    
    train_loss = train_epoch(train_loader, model, loss_fn, optimizer, scheduler=scheduler, tqdm_module=tqdm_notebook,
                             accum_steps=accum_steps, mean_accum=False, scale_loss_bs=True, loss_reduction=loss_reduction)
    val_loss = test_epoch(val_loader, model, loss_fn, loss_reduction=loss_reduction)

    val_logits = predict(model,  val_loader)
    
    if not(sigmoid_out):
        val_probs = sigmoid(val_logits)
    else:
        val_probs = val_logits
    
    if iterate_threshold:
        tvals = np.linspace(0.1, 0.5, 50)
        f2_vals = f2_threshold_wise(val_labels, val_probs, tvals=tvals)
        f2_t = np.max(f2_vals)
        best_t = tvals[np.argmax(f2_vals)]
    else:
        f2_t = fbeta_score(val_labels, binarize_prediction(val_probs, threshold=threshold), 2, average='samples')  
        best_t = threshold
    
    print('Train: {:.3f}, val: {:.3f}, f2: {:.3f}, best threshold: {:.3f}'.format(train_loss, val_loss, f2_t, best_t))

    return train_loss, val_loss, f2_t