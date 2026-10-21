from dataclasses import dataclass
from pathlib import Path

@dataclass
class PathConfig:
    rir_dir: Path
    primary_path_filename: str
    secondary_path_filename: str
    primary_path_name: str
    secondary_path_name: str
