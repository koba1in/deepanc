from dataclasses import dataclass
from pathlib import Path
import torch

@dataclass
class TrainConfig:
    device: torch.device
    batch_size: int
    delay: int

    history_path: str | None = None
    model_path: Path | None = None
    best_model_selection: bool = False
    shuffle: bool = True
    epochs: int = 30
    learning_rate: float = 0.001
    dtype: torch.dtype = torch.float32