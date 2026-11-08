from pyprojroot import here
PROJECT_ROOT = here()
DATASET_DIR = PROJECT_ROOT / "Datasets"
CSV_DIR = DATASET_DIR / "statistics"
MODEL_DIR = PROJECT_ROOT / "models"
RIR_DIR = PROJECT_ROOT / "rirs"
HISTORY_DIR = PROJECT_ROOT / "history"
print(PROJECT_ROOT)
print(MODEL_DIR)
print(RIR_DIR)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_DIR.mkdir(parents=True, exist_ok=True)



import torch
from deepanc.common.config import PathConfig
from deepanc.train.Train import train_validate
from deepanc.train.config import TrainConfig



path_config = PathConfig(
    rir_dir=RIR_DIR,
    primary_path_filename="Primary_path.mat",
    secondary_path_filename="Secondary_path.mat",
    primary_path_name="Pz1",
    secondary_path_name="S",
)

delay = 1
model_name = "customed_network_v1_freesound_delay_" + str(delay)

train_config = TrainConfig(
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    batch_size=20,
    delay=delay,
    model_path=MODEL_DIR/(model_name+".pth"),
    history_path=HISTORY_DIR/(model_name+".csv"),
    epochs=30,
    best_model_selection=False,
    )

from deepanc.common.Network import CRN, default_channels, ModelType
model = CRN(default_channels, ModelType.NONE)
train_validate(TRAIN_DATASET_BASE_FOLDER=DATASET_DIR/"train", VALIDATION_DATASET_BASE_FOLDER=DATASET_DIR/"train", TRAIN_CSV=CSV_DIR/"train.csv", VALIDATION_CSV=CSV_DIR/"validation.csv", model=model, config=train_config, path_config=path_config, parallel=True)
# train_validate(TRAIN_DATASET_BASE_FOLDER=DATASET_DIR, VALIDATION_DATASET_BASE_FOLDER=PROJECT_ROOT/"Datasets"/"train", TRAIN_CSV=None, VALIDATION_CSV=PROJECT_ROOT/"Datasets"/"statistics"/"validation.csv", config=train_config, path_config=path_config, parallel=True)
