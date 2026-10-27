import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import math
import csv

from deepanc.common.Network import crn_model
from deepanc.common.MyDataLoader import MyDataset
from deepanc.common.utils import loss, stft, spec_to_antinoise, loading_paths_from_MAT, apply_impulse_response
from deepanc.common.collate import collate_fn
        
def create_data_loader(train_data, batch_size, shuffle, device):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    
    return DataLoader(train_data, batch_size, num_workers=1, pin_memory=(device.type=="cuda"), collate_fn=collate_fn, shuffle=shuffle)

def forward_single_batch(signal, config, etas, pri_paths, sec_paths, eta_indices, path_indices, model, loss_fn):
    signal_length = signal.size(-1)
    etas = etas[eta_indices].unsqueeze(1)
    pri_paths = pri_paths[path_indices]
    sec_paths = sec_paths[path_indices]
    signal = signal.to(config.device, dtype=config.dtype)
    signal = signal.squeeze(1)
    disturbance = apply_impulse_response(signal, pri_paths)
    signal = stft(signal, config.delay)
    anti_noise = spec_to_antinoise(model(signal), sec_paths, etas, length=signal_length)
    error = disturbance - anti_noise
    loss = loss_fn(disturbance, anti_noise)
    return loss, error.square().sum().item(), disturbance.square().sum().item()

def train_single_epoch(model, train_loader, loss_fn, optimizer, config, Paths, etas,):
    model.train()
    train_loss, error_square_sum, disturbance_square_sum = 0, 0, 0
    num_samples = 0

    pri_paths = Paths[:, 0].to(config.device, dtype=config.dtype)
    sec_paths = Paths[:, 1].to(config.device, dtype=config.dtype)

    for signal in train_loader:
        optimizer.zero_grad()
        batch_size = signal.size(0)
        eta_indices = torch.randint(0, etas.shape[0], (len(signal),)).to(config.device)
        path_indices = torch.randint(0, Paths.shape[0], (len(signal), )).to(config.device)
        
        loss, e, dis = forward_single_batch(signal, config, etas, pri_paths, sec_paths, eta_indices, path_indices, model, loss_fn)

        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * batch_size
        error_square_sum += e
        disturbance_square_sum += dis

        num_samples += batch_size

    train_loss /= num_samples
    train_nmse = 10 * math.log10(error_square_sum/disturbance_square_sum)

    print(f"Training Loss: {train_loss}" + f" Training NMSE: {train_nmse}")
    return train_loss, train_nmse

def validate_single_epoch(model, validate_loader, loss_fn, config, Paths, etas, eta_indices, path_indices):
    model.eval()
    validate_loss, error_square_sum, disturbance_square_sum = 0, 0, 0
    num_samples = 0

    pri_paths = Paths[:, 0].to(config.device, dtype=config.dtype)
    sec_paths = Paths[:, 1].to(config.device, dtype=config.dtype)
    offset = 0
    with torch.no_grad():
        for signal in validate_loader:
            batch_size = signal.size(0)
            batch_indices = slice(offset, offset + batch_size)

            offset += batch_size
            loss, e, dis = forward_single_batch(signal, config, etas, pri_paths, sec_paths, eta_indices[batch_indices], path_indices[batch_indices], model, loss_fn)

            validate_loss += loss.item() * batch_size
            error_square_sum += e
            disturbance_square_sum += dis
            num_samples += batch_size

    validate_loss /= num_samples
    validate_nmse = 10 * math.log10(error_square_sum/disturbance_square_sum)

    print(f"Validate Loss: {validate_loss}" + f" Validate NMSE: {validate_nmse}")
    return validate_loss, validate_nmse
    
def train(model, train_loader, validation_loader, config, Paths, etas,):
    nmse_min_train, nmse_min = float("inf"), float("inf")
    train_loss_epochs, validate_loss_epochs = [], []
    loss_fn = loss
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=0, amsgrad=True)
    eta_indices = torch.randint(0, etas.shape[0], (len(validation_loader.dataset),)).to(config.device)
    path_indices = torch.randint(0, Paths.shape[0], (len(validation_loader.dataset),)).to(config.device)

    for i in range(config.epochs):
        print(f"Epoch {i+1}")
        print("Learning rate:", optimizer.param_groups[0]["lr"])
        train_loss, train_nmse = train_single_epoch(model, train_loader, loss_fn, optimizer, config, Paths, etas,)
        validate_loss, validate_nmse = validate_single_epoch(model, validation_loader, loss_fn, config, Paths, etas, eta_indices, path_indices)
        train_loss_epochs.append(train_loss)
        validate_loss_epochs.append(validate_loss)
        if nmse_min > validate_nmse:
            nmse_min_train, nmse_min = train_nmse, validate_nmse
            if config.best_model_selection & (config.model_path is not None):
                torch.save(model.state_dict(), config.model_path)
                print("Trained feed forward net saved at " + str(config.model_path)) 
        print("------------------------------")

        if config.history_path is not None:
            with open(config.history_path, mode="a", encoding="utf-8", newline="", ) as file:
                writer = csv.writer(file)
                writer.writerow([train_loss, validate_loss, train_nmse, validate_nmse])

    if not config.best_model_selection:
        torch.save(model.state_dict(), config.model_path)
        print("model saved at " + str(config.model_path))

    print("Finished training")
    return train_loss_epochs, nmse_min_train, validate_loss_epochs, nmse_min

def  train_validate(TRAIN_DATASET_BASE_FOLDER, VALIDATION_DATASET_BASE_FOLDER, TRAIN_CSV, VALIDATION_CSV, config, path_config, parallel=True):

    model = crn_model
    gpus = torch.cuda.device_count() if (config.device.type == "cuda") else 0
    use_parallel = parallel and gpus > 1
    num_devices = gpus if use_parallel else 1
    
    if use_parallel:
        print(f"gpu: {gpus}")
        model = torch.nn.DataParallel(model,)

    effective_batch_size = config.batch_size * num_devices

    if config.history_path is not None:
        config.history_path.touch()
        with open(config.history_path, mode="w", encoding="utf-8", newline="", ) as file:
            writer = csv.writer(file)
            writer.writerow(["train_loss", "validate_loss", "train_nmse", "validate_nmse"])

    train_data = MyDataset(TRAIN_DATASET_BASE_FOLDER, TRAIN_CSV)
    validate_data = MyDataset(VALIDATION_DATASET_BASE_FOLDER, VALIDATION_CSV)
    train_loader = create_data_loader(train_data, effective_batch_size, shuffle=config.shuffle, device=config.device)
    validate_dataloader = create_data_loader(validate_data, effective_batch_size, shuffle=False, device=config.device)
    rirs = path_config.rir_dir.iterdir()
    paths = [torch.tensor(loading_paths_from_MAT(path_config.rir_dir, rir, path_config.primary_path_filename, path_config.secondary_path_filename, path_config.primary_path_name, path_config.secondary_path_name), dtype=config.dtype).unsqueeze(1) for rir in rirs]
    paths = torch.stack(paths)
    paths = paths.to(config.device)
    # paths: [rirs, 2, 1, numtaps]

    model = model.to(config.device, dtype=config.dtype)
    etas = torch.tensor([math.sqrt(0.1), math.sqrt(1), math.sqrt(10), torch.finfo(config.dtype).max], dtype=config.dtype).to(config.device)
    train_loss_epochs, nmse_min_train, validate_loss_epochs, nmse_min = train(model, train_loader, validate_dataloader, config, paths, etas,)
    return train_loss_epochs, nmse_min_train, validate_loss_epochs, nmse_min

