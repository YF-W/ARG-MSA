"""
AMIO -- All Model in One
"""
import importlib

import torch.nn as nn

from .subNets.AlignNets import AlignSubNet

__all__ = ["AMIO"]


MODEL_IMPORTS = {
    "tfn": (".singleTask.TFN", "TFN"),
    "arg_msa": (".ARG_MSA", "ARG_MSA"),
}


def _load_model_class(model_name: str):
    if model_name not in MODEL_IMPORTS:
        raise KeyError(f"Unknown model name: {model_name}")
    module_name, class_name = MODEL_IMPORTS[model_name]
    module = importlib.import_module(module_name, package=__package__)
    return getattr(module, class_name)


class AMIO(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.need_model_aligned = args.get("need_model_aligned", None)
        if self.need_model_aligned:
            self.alignNet = AlignSubNet(args, "avg_pool")
            if "seq_lens" in args.keys():
                args["seq_lens"] = self.alignNet.get_seq_len()

        model_name = args["model_name"]
        model_cls = _load_model_class(model_name)

        self.Model = model_cls(args)

    def forward(self, text_x, audio_x, video_x, *args, **kwargs):
        if self.need_model_aligned:
            text_x, audio_x, video_x = self.alignNet(text_x, audio_x, video_x)
        return self.Model(text_x, audio_x, video_x, *args, **kwargs)
