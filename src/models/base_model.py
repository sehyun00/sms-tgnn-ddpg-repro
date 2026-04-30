from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple


class BaseModel(nn.Module, ABC):
    """
    Abstract Base Class for all models.
    Enforces a standard interface for training and inference.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.device = torch.device(
            config["project"]["device"] if torch.cuda.is_available() else "cpu"
        )

    @abstractmethod
    def forward(self, x: Any) -> Any:
        """Standard PyTorch forward pass."""
        pass

    @abstractmethod
    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Inference method.
        Should handle moving data to device and returning detached tensor (cpu).
        """
        pass

    def save(self, path: str):
        """Saves model state dict."""
        torch.save(self.state_dict(), path)

    def load(self, path: str, strict: bool = True):
        """
        Loads model state dict.
        If strict=False, ignores keys with shape mismatches (Partial Loading).
        """
        loaded_state_dict = torch.load(path, map_location=self.device)

        if not strict:
            model_state_dict = self.state_dict()
            filtered_state_dict = {}
            skipped_layers = []

            for k, v in loaded_state_dict.items():
                if k in model_state_dict:
                    if v.shape == model_state_dict[k].shape:
                        filtered_state_dict[k] = v
                    else:
                        skipped_layers.append(k)
                else:
                    # Key not in model (unexpected key), skip if not strict
                    pass

            if skipped_layers:
                print(
                    f"      ⚠️ Partial Loading: Skipped {len(skipped_layers)} layers due to shape mismatch:"
                )
                for k in skipped_layers[:3]:  # Show first 3 only
                    print(f"         - {k}")
                if len(skipped_layers) > 3:
                    print(f"         ... and {len(skipped_layers) - 3} more.")

            # Load with strict=False to allow missing keys (since we filtered some out)
            self.load_state_dict(filtered_state_dict, strict=False)
        else:
            self.load_state_dict(loaded_state_dict, strict=True)
