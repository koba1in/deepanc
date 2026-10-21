from pyprojroot import here
PROJECT_ROOT = here()
DATASET_DIR = PROJECT_ROOT / "Dataset"
MODEL_DIR = PROJECT_ROOT / "models"
RIR_DIR = PROJECT_ROOT / "rirs"
print(PROJECT_ROOT)
print(MODEL_DIR)
print(RIR_DIR)
MODEL_DIR.mkdir(parents=True, exist_ok=True)



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
model_name = "delay_" + str(delay) + "v3.pth"

train_config = TrainConfig(
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    batch_size=25,
    delay=delay,
    model_path=MODEL_DIR/model_name,
    )




train_validate(TRAIN_DATASET_BASE_FOLDER=DATASET_DIR/"train", VALIDATION_DATASET_BASE_FOLDER=DATASET_DIR/"validate", config=train_config, path_config=path_config, parallel=True)