from common.utils import loss, nmse, stft, spec_to_antinoise, loading_paths_from_MAT, apply_impulse_response, check_finite
import torch
import os
from torch.utils.data import DataLoader
import torch.optim as optim
from common.Network import crn_model
from common.MyDataLoader import MyDataset

BATCH_SIZE = 250
EPOCHS = 30

def init_weights(m):
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        
def create_data_loader(train_data, batch_size, shuffle, device):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return DataLoader(train_data, batch_size, num_workers=2, pin_memory=(device.type=="cuda"), shuffle=shuffle)
    
def train_single_epoch(model, data_loader, loss_fn, optimizer, device, Paths, delay, ETAS, dtype):
    model.train()
    train_loss = 0
    train_nmse = 0
    PRI_PATHS = Paths[:, 0].to(device, dtype=dtype)
    SEC_PATHS = Paths[:, 1].to(device, dtype=dtype)
    # torch.autograd.set_detect_anomaly(True)
    for signal in data_loader:
        eta_indices = torch.randint(0, ETAS.shape[0], (len(signal),)).to(device)
        path_indices = torch.randint(0, Paths.shape[0], (len(signal), )).to(device)
        etas = ETAS[eta_indices].unsqueeze(1)
        pri_paths = PRI_PATHS[path_indices]
        sec_paths = SEC_PATHS[path_indices]
        signal = signal.to(device, dtype=dtype)
        signal = signal.squeeze(1)
        disturbance = apply_impulse_response(signal, pri_paths)
        signal = stft(signal, delay)
        anti_noise = spec_to_antinoise(model(signal), sec_paths, etas)
        error = disturbance - anti_noise
        loss = loss_fn(disturbance, anti_noise)
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item()
        train_nmse += torch.pow(10, nmse(error, disturbance)/10).item()
    train_loss /= len(data_loader)
    train_nmse /= len(data_loader)
    train_nmse = 10 * torch.log10(torch.tensor(train_nmse)).item()
    
    print(f"Training Loss: {train_loss}" + f" Training NMSE: {train_nmse}")
    return train_loss, train_nmse

def validate_single_epoch(model, data_loader, loss_fn, device, Paths, delay, ETAS,dtype, eta_indices, path_indices):
    model.eval()
    validate_loss = 0
    validate_nmse = 0
    PRI_PATHS = Paths[:, 0].to(device, dtype=dtype)
    SEC_PATHS = Paths[:, 1].to(device, dtype=dtype)
    offset = 0
    with torch.no_grad():
        for signal in data_loader:
            batch_size = signal.size(0)
            batch_indices = slice(offset, offset + batch_size)
            etas = ETAS[eta_indices[batch_indices]].unsqueeze(1)
            pri_paths = PRI_PATHS[path_indices[batch_indices]]
            sec_paths = SEC_PATHS[path_indices[batch_indices]]
            offset += batch_size
            signal = signal.to(device, dtype=dtype)
            signal = signal.squeeze(1)
            disturbance = apply_impulse_response(signal, pri_paths)
            signal = stft(signal, delay)
            anti_noise = spec_to_antinoise(model(signal), sec_paths, etas)
            error = disturbance - anti_noise
            loss = loss_fn(disturbance, anti_noise)
            validate_loss += loss.item()
            validate_nmse += torch.pow(10, nmse(error, disturbance)/10).item()
        
    validate_loss /= len(data_loader)
    validate_nmse /= len(data_loader)
    validate_nmse = 10 * torch.log10(torch.tensor(validate_nmse)).item()
    
    print(f"Validate Loss: {validate_loss}" + f" Validate NMSE: {validate_nmse}")
    return validate_loss, validate_nmse
    
def train(model, data_loader, eva_data_loader, epochs, device, Paths, delay, ETAS, dtype, MODEL_PTH=None,):
    nmse_min = float("inf")
    nmse_min_train = float("inf")
    loss_fn = loss
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0, amsgrad=True) # L2 regularization
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # reduce the learning after 5 epochs
    train_loss_epochs = []
    validate_loss_epochs = []
    eta_indices = torch.randint(0, ETAS.shape[0], (len(eva_data_loader.dataset),)).to(device)
    path_indices = torch.randint(0, Paths.shape[0], (len(eva_data_loader.dataset),)).to(device)
    for i in range(epochs):
        print(f"Epoch {i+1}")
        print("Learning rate:", optimizer.param_groups[0]["lr"])
        train_loss, train_nmse = train_single_epoch(model, data_loader, loss_fn, optimizer, device, Paths, delay, ETAS, dtype)
        validate_loss, validate_nmse = validate_single_epoch(model, eva_data_loader, loss_fn, device, Paths, delay, ETAS, dtype, eta_indices, path_indices)
        # scheduler.step()
        train_loss_epochs.append(train_loss)
        validate_loss_epochs.append(validate_loss)
        if nmse_min > validate_nmse:
            nmse_min = validate_nmse
            nmse_min_train = train_nmse
            if MODEL_PTH is not None:
                torch.save(model.state_dict(), MODEL_PTH)
            print("Trained feed forward net saved at " + str(MODEL_PTH)) 
        print("------------------------------")
    print("Finished training")
    return train_loss_epochs, nmse_min_train, validate_loss_epochs, nmse_min
import math
def train_validate(TRAIN_DATASET_BASE_FOLDER, VALIDATION_DATASET_BASE_FOLDER, MODEL_PTH, rirsdir, Pri_Path, Sec_Path, delay, Pri_path_name="Pz1", Sec_path_name="S", batch_size=100, epochs=None, parallel=True):
    dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = crn_model
    gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    use_parallel = parallel and gpus > 1
    num_devices = gpus if use_parallel else 1
    

    if use_parallel:
        print(f"gpu: {gpus}")
        model = torch.nn.DataParallel(model,)

    effective_batch_size = batch_size * num_devices
    train_data = MyDataset(TRAIN_DATASET_BASE_FOLDER)
    validate_data = MyDataset(VALIDATION_DATASET_BASE_FOLDER)
    data_loader = create_data_loader(train_data, effective_batch_size, shuffle=True, device=device)
    validate_dataloader = create_data_loader(validate_data, effective_batch_size, shuffle=False, device=device)
    
    rirs = os.listdir(rirsdir)
    paths = [torch.tensor(loading_paths_from_MAT(rirsdir, rir, Pri_Path, Sec_Path, Pri_path_name, Sec_path_name), dtype=dtype).unsqueeze(1) for rir in rirs]
    paths = torch.stack(paths)
    paths = paths.to(device)
    # paths: [rirs, 2, 1, numtaps]


    model = model.to(device, dtype=dtype)
    ETAS = torch.tensor([math.sqrt(0.1), math.sqrt(1), math.sqrt(10), torch.finfo(dtype).max], dtype=dtype).to(device)
    if epochs is None:
        epochs = EPOCHS
    train_loss_epochs, nmse_min_train, validate_loss_epochs, nmse_min = train(model, data_loader, validate_dataloader, epochs, device, paths, delay, ETAS, dtype, MODEL_PTH)
    return train_loss_epochs, nmse_min_train, validate_loss_epochs, nmse_min

