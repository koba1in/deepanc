from dataclasses import dataclass
from pathlib import Path
import torch

@dataclass
class TrainConfig:
    device: torch.device
    batch_size: int
    delay: int

    model_path: Path | None = None
    epochs: int = 30
    learning_rate: float = 0.001
    dtype: torch.dtype = torch.float32