"""
ATIO -- All Trains in One
"""
import importlib

__all__ = ["ATIO"]


TRAIN_IMPORTS = {
    "tfn": (".singleTask.TFN", "TFN"),
    "arg_msa": (".ARG_MSA", "ARG_MSA"),
}


def _load_trainer_class(model_name: str):
    if model_name not in TRAIN_IMPORTS:
        raise KeyError(f"Unknown trainer name: {model_name}")
    module_name, class_name = TRAIN_IMPORTS[model_name]
    module = importlib.import_module(module_name, package=__package__)
    return getattr(module, class_name)


class ATIO:
    def getTrain(self, args):
        trainer_cls = _load_trainer_class(args["model_name"])
        return trainer_cls(args)
