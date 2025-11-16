

from abc import abstractmethod
from pathlib import Path

import torch
import torch.nn as nn

from ultralytics.utils import checks
from ultralytics.utils.torch_utils import smart_inference_mode

try:
    import clip
except ImportError:
    checks.check_requirements("git+https://github.com/ultralytics/CLIP.git")
    import clip


class TextModel(nn.Module):
    

    def __init__(self):
        
        super().__init__()

    @abstractmethod
    def tokenize(texts):
        
        pass

    @abstractmethod
    def encode_text(texts, dtype):
        
        pass


class CLIP(TextModel):
    

    def __init__(self, size, device):
        
        super().__init__()
        self.model = clip.load(size, device=device)[0]
        self.to(device)
        self.device = device
        self.eval()

    def tokenize(self, texts):
        
        return clip.tokenize(texts).to(self.device)

    @smart_inference_mode()
    def encode_text(self, texts, dtype=torch.float32):
        
        txt_feats = self.model.encode_text(texts).to(dtype)
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        return txt_feats


class MobileCLIP(TextModel):
    

    config_size_map = {"s0": "s0", "s1": "s1", "s2": "s2", "b": "b", "blt": "b"}

    def __init__(self, size, device):
        
        try:
            import warnings


            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                import mobileclip
        except ImportError:

            checks.check_requirements("git+https://github.com/ultralytics/mobileclip.git")
            import mobileclip

        super().__init__()
        config = self.config_size_map[size]
        file = f"mobileclip_{size}.pt"
        if not Path(file).is_file():
            from ultralytics import download

            download(f"https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/{file}")
        self.model = mobileclip.create_model_and_transforms(f"mobileclip_{config}", pretrained=file, device=device)[0]
        self.tokenizer = mobileclip.get_tokenizer(f"mobileclip_{config}")
        self.to(device)
        self.device = device
        self.eval()

    def tokenize(self, texts):
        
        return self.tokenizer(texts).to(self.device)

    @smart_inference_mode()
    def encode_text(self, texts, dtype=torch.float32):
        
        text_features = self.model.encode_text(texts).to(dtype)
        text_features /= text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features


class MobileCLIPTS(TextModel):
    

    def __init__(self, device):
        
        super().__init__()
        from ultralytics.utils.downloads import attempt_download_asset

        self.encoder = torch.jit.load(attempt_download_asset("mobileclip_blt.ts"), map_location=device)
        self.tokenizer = clip.clip.tokenize
        self.device = device

    def tokenize(self, texts):
        
        return self.tokenizer(texts).to(self.device)

    @smart_inference_mode()
    def encode_text(self, texts, dtype=torch.float32):
        
        return self.encoder(texts)


def build_text_model(variant, device=None):
    
    base, size = variant.split(":")
    if base == "clip":
        return CLIP(size, device)
    elif base == "mobileclip":
        return MobileCLIPTS(device)
    else:
        raise ValueError(f"Unrecognized base model: '{base}'. Supported base models: 'clip', 'mobileclip'.")
