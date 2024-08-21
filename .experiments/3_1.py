import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import os, sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools

module_path = '/home/sbose/time-series-forecasting-federation'
FOLDER_NAME = 'centralized_results'
        
# configure model and other stuff
lookahead = 4
dtype = torch.float32
device = 'cuda'
num_clients = 12

# Contingent imports
sys.path.insert(0,module_path)
from files_for_appfl.comstock_dataloader import get_comstock_shared_norm, get_comstock_range
from files_for_appfl.loss_last import MSELoss
from files_for_appfl.metric import mape
from models.LSTM.LSTMAR import LSTMAR
from models.DARNN.DARNN import DARNN
from models.TRANSFORMER.Transformer import Transformer

# zero weight init
def zero_weights(model):
    for param in model.parameters():
        param.data.zero_()
        
# function to zero the weights for initialization
def normal_weights(model):
    for param in model.parameters():
        param.data.normal_()
        
# function to calculate norm of gradients
def calculate_gradient_norm(model):
    total_norm = 0
    for param in model.parameters():
        param_norm = param.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

# master function for training
def train_and_test_transformer(
optim_name, # pass as the name containe in a string
custom_str = 'Transformer, FullFeatureSet, LongTrain',
normalize = 'True',
display_time_idx = 500,
device = 'cuda' if torch.cuda.is_available() else 'cpu',
seed = 42,
BS = 32,
steps = 1000,
clip_grad = np.inf,
ntype = 'z',
xformer_dim = 128,
lr = 1e-5,
test_every = 100,
lookback = 12
):
    
    # master function to train on data and produce output on test set 
    model_kwargs = {
        'x_size': 6,
        'y_size': 1,
        'u_size': 2,
        's_size': 7,
        'lookback': lookback,
        'lookahead': lookahead,
        'd_model' : xformer_dim,
        'e_layers' : 4,
        'd_layers' : 4,
        'dtype' : dtype
    }
    model = nn.DataParallel(Transformer(**model_kwargs))
    model = model.to(device)
    
    optim = eval(optim_name)(model.parameters(), **{'lr':lr})
    loss_fn = MSELoss(ntype)
    loss_fn_to_report = MSELoss(ntype)
    
    # get and combine datasets
    _, train_set, test_set = get_comstock_range(
        end_bldg_idx=num_clients,
        lookback = lookback,
        lookahead = lookahead,
        dtype = dtype,
        normalize = normalize,
        normalize_type=ntype
    )
    train_set, test_set = ConcatDataset(train_set), ConcatDataset(test_set)
    torch.manual_seed(seed)
    train_loader = DataLoader(train_set, batch_size=BS, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=8096, shuffle=True)

    loss_record, mape_record, norm_record = [], [], []
    elapsed = 0
    normal_weights(model) # actually initializes to normal, doesnt zero   
    for inp, lab in (t:=tqdm(itertools.cycle(train_loader))):
    
        inp, lab = inp.to(device), lab.to(device)
        pred = model(inp)
        loss = loss_fn(lab,pred)
        loss_to_report = loss_fn_to_report(lab,pred)
        optim.zero_grad()
        loss.backward()
        if not np.isinf(clip_grad) and clip_grad > 0:
            # clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        norm_record.append(calculate_gradient_norm(model))
        optim.step()
        # scheduler.step()    
        loss_record.append(loss_to_report.item())
        elapsed += 1
        
        t.set_description(f"On experiment {custom_str}, step {elapsed}, loss is {loss.item()}.")
        
        if elapsed % test_every == 0:
            mapes = []
            for inp,lab in test_loader:
                inp = inp.to(device)
                with torch.no_grad():
                    pred = model(inp)
                mapes.append(mape(lab.to('cpu').numpy(),pred.to('cpu').numpy(),normalization_type=ntype))
            metric = np.mean(np.array(mapes))
            mape_record.append(metric)
            print(f"On step {elapsed}, MAPE error is {metric} percent.")
            
        if elapsed == steps:
            break
    
    # plotting here
    fig, axs = plt.subplots(1, 4, figsize=(20,4))
    # plot losses
    loss_record = np.array(loss_record)
    axs[0].plot(np.arange(1,loss_record.size+1),np.array(loss_record))
    axs[0].set_xlim(1,loss_record.size)
    axs[0].set_xlabel('Steps')
    axs[0].set_ylabel('MSE Loss')
    axs[0].set_title(f'Train Loss')
    axs[0].set_yscale('log')
    # plot norms
    norm_record = np.array(norm_record)
    axs[1].plot(np.arange(1,norm_record.size+1),np.array(norm_record))
    axs[1].set_xlim(1,norm_record.size)
    axs[1].set_xlabel('Steps')
    axs[1].set_ylabel('L2 Norm')
    axs[1].set_title(f'Gradient Norm clip: {clip_grad}')
    axs[1].set_yscale('log')
    # plot MAPEs
    mape_record = np.array(mape_record)
    axs[2].plot(np.arange(1,mape_record.size+1),np.array(mape_record),'ko-')
    axs[2].set_xlim(1,mape_record.size)
    axs[2].set_xlabel(f'Steps x{test_every}')
    axs[2].set_ylabel(f'MAPE')
    axs[2].set_title(f'Test set.')
    # plot the test sets
    inputs, outputs = [], []
    for idx in range(display_time_idx):
        itm = test_set.__getitem__(idx)
        inputs.append(itm[0])
        outputs.append(itm[1].numpy())
    batched_input = torch.stack(inputs).to(dtype).to(device)
    with torch.no_grad():   
        batched_output = model(batched_input).to('cpu').numpy()
    preds = list(batched_output)
    plot_gt, plot_pred = [], []
    for idx in range(display_time_idx):
        minval, maxval = outputs[idx][-1,1], outputs[idx][-1,2]
        if ntype == 'minmax':
            # minmax
            plot_gt.append((outputs[idx][-1,0]-minval)/(maxval-minval))
        else:
            # z normalization
            plot_gt.append((outputs[idx][-1,0]-minval)/maxval)
        plot_pred.append(preds[idx][-1,0])
    plot_gt, plot_pred = np.array(plot_gt), np.array(plot_pred)
    axs[3].plot(np.arange(1,plot_gt.size+1),plot_gt,label='ground truth')
    axs[3].plot(np.arange(1,plot_pred.size+1),plot_pred,label='prediction')
    axs[3].set_xlim(1,plot_pred.size)
    axs[3].set_xlabel('Time index')
    axs[3].set_ylabel('kWh')
    axs[3].legend()
    axs[3].set_title(f'Reconstruction')
    
    optdict = {
        'torch.optim.SGD': 'sgd',
        'torch.optim.Adam': 'adam'
    }
    
    plt.suptitle(f'Optim={optdict[optim_name]}, BS={BS}, lr={lr}, clip={clip_grad}')
    
    plt.savefig(f'/home/sbose/{FOLDER_NAME}/{optdict[optim_name]}_BS_{BS}_lr_{lr}_lookback_{lookback}_transformer_{steps}.pdf',format='pdf',bbox_inches='tight')
    plt.close()
    # torch.save(model.state_dict(),f'/home/sbose/{FOLDER_NAME}/{optdict[optim_name]}_BS_{BS}_lr_{lr}_clip_{clip_grad}.pth')
    np.savez_compressed(f'/home/sbose/{FOLDER_NAME}/{optdict[optim_name]}_BS_{BS}_lr_{lr}_lookback_{lookback}_transformer_{steps}.npz',loss_record=loss_record, mape_record=mape_record,norm_record=norm_record,preds=preds,outputs=outputs)
    
    return None

if __name__ == "__main__":
    
    configs = [
        # [1e-4, 128, 100, 'torch.optim.SGD', 12, 20000],
        # [1e-5, 128, 100, 'torch.optim.SGD', 12, 20000],
        # [1e-4, 128, 100, 'torch.optim.SGD', 12, 100000],
        [1e-4, 128, 100, 'torch.optim.SGD', 12, 200000]
    ]

    os.makedirs(f'/home/sbose/{FOLDER_NAME}',exist_ok=True)
    file = open(f'/home/sbose/{FOLDER_NAME}/done.txt','a')

    for l, b, c, o, lb, stp in configs:
        
        train_and_test_transformer(
            o, # pass as the name containe in a string
            'Transformer',
            normalize=True,
            display_time_idx=250,
            ntype = 'z',
            xformer_dim=128,
            BS = b,
            lr = l,
            steps=stp,
            test_every = 6000,
            clip_grad= c,
            lookback = lb
        )
        print(f"Finished opt={o}, lr={l}, BS={b}, Clip={c}.")
        file.write(f"'{o}',{l},{b},{c}\n")
        
    file.close()