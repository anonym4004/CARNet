from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Solar, Dataset_PEMS
from torch.utils.data import DataLoader



import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import random
import os

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


set_seed(2024)

from TCF.model.CARNet import Model

size = [96,48,720]
d_model = 128
d_ff = 128
d_core = 64
e_layers = 1
bs = 16
lr = 0.001
n_vars = 321
patch_len = 16
period_len = 24
cycle_len = 168
freq = 'h'
use_norm = 1
embed = 'timeF'
root_path = 'C:/Users/Awsftausif/Desktop/S-Mamba_datasets/electricity/'
data_path = 'electricity.csv'
train_set = Dataset_Custom(
    root_path=root_path,
    data_path=data_path,
    flag='train',
    size=size,
    features='M',      # 'M' = multivariate (use all features)
    target='OT',  # change this to your target column
    scale=True,
    timeenc=0,
    freq=freq,           # depends on your dataset frequency (h=hourly, d=daily, etc.)
    cycle=cycle_len
)

val_set = Dataset_Custom(
    root_path=root_path,
    data_path=data_path,
    flag='val',
    size=size,
    features='M',
    target='OT',
    scale=True,
    timeenc=0,
    freq=freq,
    cycle=cycle_len
)

test_set = Dataset_Custom(
    root_path=root_path,
    data_path=data_path,
    flag='test',
    size=size,
    features='M',
    target='OT',
    scale=True,
    timeenc=0,
    freq=freq,
    cycle=cycle_len
)


train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, drop_last=True)
val_loader = DataLoader(val_set, batch_size=bs, shuffle=False, drop_last=True)
test_loader = DataLoader(test_set, batch_size=bs, shuffle=False, drop_last=False)


def find_period(x, k=1):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0:2] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period

for batch_idx, (seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index) in enumerate(train_loader):
    print("Batch index:", batch_idx)
    
    # Print the first sample in this batch
    print("seq_x shape:", seq_x.shape)
    
    print("\nseq_y shape:", seq_y.shape)
    
    print("\nseq_x_mark shape:", seq_x_mark.shape)
    
    print("\nseq_y_mark shape:", seq_y_mark.shape)
    
    period = find_period(seq_x, k=1)

    print("Detected period:", period)
    
    print("\nCycle index: ", cycle_index)
    
    # Exit after first batch
    break



class Config:
    def __init__(self):
        self.d_model = d_model
        self.d_core = d_core
        self.e_layers = e_layers
        self.d_ff = d_ff
        self.n_vars = n_vars
        self.enc_in = n_vars
        self.patch_len = patch_len 
        self.cycle_len = cycle_len
        self.seq_len = size[0]
        self.pred_len = size[2]
        self.kernel_size = 0
        self.n_heads = 8
        self.factor = 3
        self.dropout = 0
        self.use_norm = use_norm
        self.period_len = period_len
        self.freq = freq
        self.embed = embed
        self.task_name = 'long_term_forecast'
        self.features = 'M'
        self.activation = 'gelu'
# Create configuration instance
configs = Config()
model = Model(configs).to('cuda')



# ---- Loss & Metrics ----
mse_loss = nn.MSELoss()

def mae_loss(pred, true):
    return torch.mean(torch.abs(pred - true))

# ---- Train & Evaluate ----
def train(model, train_loader, optimizer, scheduler, device, pred_len):
    model.train()
    total_loss = 0

    for seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index in tqdm(
            train_loader, desc="Training", leave=False):

        seq_x = seq_x.to(device).float()
        target = seq_y[:, -pred_len:, :].to(device).float()

        optimizer.zero_grad()
        outputs = model(seq_x, cycle_index)
        loss = mse_loss(outputs, target)

        loss.backward()
        optimizer.step()
        scheduler.step()   

        total_loss += loss.item()

    return total_loss / len(train_loader)

def evaluate(model, val_loader, device, pred_len):
    model.eval()
    total_mse, total_mae = 0, 0
    with torch.no_grad():
        for seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index in tqdm(val_loader, desc="Validating", leave=False):
            seq_x = seq_x.to(device).float()
            target = seq_y[:, -pred_len:, :].to(device).float()

            outputs = model(seq_x, cycle_index)
            total_mse += mse_loss(outputs, target).item()
            total_mae += mae_loss(outputs, target).item()

    return total_mse / len(val_loader), total_mae / len(val_loader)


def test(model, test_loader, device, pred_len):
    model.eval()
    total_mse, total_mae = 0, 0
    with torch.no_grad():
        for seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index in tqdm(test_loader, desc="Testing", leave=False):
            seq_x = seq_x.to(device).float()
            target = seq_y[:, -pred_len:, :].to(device).float()

            outputs = model(seq_x, cycle_index)
            total_mse += mse_loss(outputs, target).item()
            total_mae += mae_loss(outputs, target).item()
    return total_mse / len(test_loader), total_mae / len(test_loader), None, None


# ---- Main Training Loop ----
def train_model(
        model,
        train_loader,
        val_loader,
        test_loader,
        pred_len,
        epochs=100,
        lr=lr,
        patience=5,
        device='cuda'
):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    steps_per_epoch = len(train_loader)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=lr,                  # same idea as first code
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,              # warmup fraction (typical)
    )

    best_val_loss = float('inf')
    best_model = None
    patience_counter = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        train_loss = train(
            model, train_loader, optimizer, scheduler, device, pred_len
        )

        val_mse, val_mae = evaluate(
            model, val_loader, device, pred_len
        )

        print(
            f"Train Loss: {train_loss:.6f} | "
            f"Val MSE: {val_mse:.6f} | "
            f"Val MAE: {val_mae:.6f}"
        )

        # ---- Early stopping ----
        if val_mse < best_val_loss:
            best_val_loss = val_mse
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Load best model
    model.load_state_dict(best_model)

    # Final test
    test_mse, test_mae, preds, trues = test(
        model, test_loader, device, pred_len
    )

    print(f"\nTest MSE: {test_mse:.6f} | Test MAE: {test_mae:.6f}")
    return model, preds, trues



# ---- Run training ----
train_model(model, train_loader, val_loader, test_loader, pred_len=size[2])
