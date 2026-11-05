"""Train or evaluate ARG_MSA and TFN using the original regression trainers."""
import argparse
import gc
import json
import logging
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import get_config_regression
from data_loader import MMDataLoader
from models import AMIO
from trains import ATIO
from utils import assign_gpu, count_parameters, setup_seed

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
SUPPORTED_MODELS = ("arg_msa", "tfn")
SUPPORTED_DATASETS = ("mosi", "mosei", "sims", "simsv2")
logger = logging.getLogger("MSA")
formatted_now = datetime.now().strftime("%Y%m%d_%H%M%S")
SUMMARY_COLUMNS = ["epoch", "Val_Acc_7", "Val_Acc_5", "Val_Acc_3", "Val_Acc_2",
                   "Val_F1_score", "Val_Corr", "Val_MAE", "Train_Loss", "Val_Loss"]


def _safe_float(value, field_name, epoch_idx):
    if torch.is_tensor(value):
        value = value.detach().cpu().item()
    try:
        out = float(value)
    except Exception as ex:
        raise RuntimeError(f"[CSVGuard] {field_name} cannot cast to float at epoch={epoch_idx}: {value}") from ex
    if not np.isfinite(out):
        raise RuntimeError(f"[CSVGuard] {field_name} is non-finite at epoch={epoch_idx}: {out}")
    return out

def _safe_summary_metric(value, field_name, epoch_idx):
    """
    Keep CSVGuard strict for core training signals such as Loss/MAE, but allow a
    controlled fallback for correlation metrics when variance collapses and
    numpy/sklearn returns nan.
    """
    try:
        return _safe_float(value, field_name, epoch_idx)
    except RuntimeError as ex:
        if "non-finite" not in str(ex):
            raise
        if "corr" not in field_name.lower():
            raise
        logger.warning(
            "[CSVGuard] %s is non-finite at epoch=%s, fallback to 0.0 for summary csv.",
            field_name,
            epoch_idx,
        )
        return 0.0

def _audit_csv_file(csv_file: Path, expected_rows: int):
    df = pd.read_csv(csv_file)
    logger.info("[CSVAudit] file=%s shape=%s columns=%s", csv_file, df.shape, list(df.columns))
    logger.info("[CSVAudit] dtypes=%s", {c: str(t) for c, t in df.dtypes.items()})

    if list(df.columns) != SUMMARY_COLUMNS:
        raise RuntimeError(f"[CSVAudit] columns mismatch: got={list(df.columns)} expect={SUMMARY_COLUMNS}")
    if len(df) != int(expected_rows):
        raise RuntimeError(f"[CSVAudit] row count mismatch: got={len(df)} expect={expected_rows}")

    for col in SUMMARY_COLUMNS:
        series = pd.to_numeric(df[col], errors='coerce')
        bad_mask = series.isna() | ~np.isfinite(series.values)
        if bad_mask.any():
            bad_rows = df.index[bad_mask].tolist()
            logger.error("[CSVAudit] non-numeric rows in col=%s rows=%s", col, bad_rows)
            for idx in bad_rows:
                logger.error("[CSVAudit] row[%s]=%s", idx, df.loc[idx].to_dict())
            raise RuntimeError(f"[CSVAudit] non-numeric value in {col}: rows={bad_rows}")

    epoch_num = pd.to_numeric(df['epoch'], errors='coerce')
    if ((epoch_num % 1) != 0).any():
        bad_rows = df.index[((epoch_num % 1) != 0)].tolist()
        raise RuntimeError(f"[CSVAudit] epoch not integer rows={bad_rows}")

    logger.info("[CSVAudit] passed: summary csv numeric/integer checks all passed.")

def _pick_metric(metric_dict, field_aliases, save_field, epoch_idx):
    for key in field_aliases:
        if key in metric_dict:
            return _safe_summary_metric(metric_dict[key], save_field, epoch_idx)
    raise RuntimeError(f"[CSVGuard] missing metric {save_field} at epoch={epoch_idx}, keys={list(metric_dict.keys())}")

def _build_summary_rows(epoch_results):
    rows = []
    num_epochs = len(epoch_results['train'])
    for epoch in range(num_epochs):
        train_epoch = epoch_results['train'][epoch]
        valid_epoch = epoch_results['valid'][epoch]

        if 'TrainBatches' in train_epoch and int(train_epoch['TrainBatches']) <= 0:
            raise RuntimeError(f"[CSVGuard] TrainBatches <= 0 at epoch={epoch + 1}")

        train_loss = _safe_float(train_epoch.get('LossMean', train_epoch.get('Loss')), 'Train_Loss', epoch + 1)
        row = {
            'epoch': int(epoch + 1),
            'Val_Acc_7': _pick_metric(valid_epoch, ('Acc_7', 'acc_7'), 'Val_Acc_7', epoch + 1),
            'Val_Acc_5': _pick_metric(valid_epoch, ('Acc_5', 'acc_5'), 'Val_Acc_5', epoch + 1),
            'Val_Acc_3': _pick_metric(valid_epoch, ('Acc_3', 'acc_3'), 'Val_Acc_3', epoch + 1),
            'Val_Acc_2': _pick_metric(valid_epoch, ('Acc_2', 'acc_2'), 'Val_Acc_2', epoch + 1),
            'Val_F1_score': _pick_metric(valid_epoch, ('F1_score', 'f1_score'), 'Val_F1_score', epoch + 1),
            'Val_Corr': _pick_metric(valid_epoch, ('Corr', 'corr'), 'Val_Corr', epoch + 1),
            'Val_MAE': _pick_metric(valid_epoch, ('MAE', 'mae'), 'Val_MAE', epoch + 1),
            'Train_Loss': train_loss,
            'Val_Loss': _safe_float(valid_epoch.get('Loss'), 'Val_Loss', epoch + 1),
        }
        rows.append(row)
    return rows

def _set_logger(log_dir, model_name, dataset_name, verbose_level):

    # base logger
    log_file_path = Path(log_dir) / f"{model_name}-{dataset_name}-{formatted_now}.log"
    logger = logging.getLogger('MSA') 
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    # file handler
    fh = logging.FileHandler(log_file_path)
    fh_formatter = logging.Formatter('%(asctime)s - %(name)s [%(levelname)s] - %(message)s')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    # stream handler
    stream_level = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG}
    ch = logging.StreamHandler()
    ch.setLevel(stream_level[verbose_level])
    ch_formatter = logging.Formatter('%(name)s - %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger

def _close_logger_handlers():
    logger = logging.getLogger('MSA')
    for handler in logger.handlers[:]:
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


def _json_value(value):
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, (Path, torch.device)):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _save_evaluation(args, results):
    output_dir = Path(args.res_save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.model_name}_{args.dataset_name}_test"
    metrics = {k: v for k, v in results.items()
               if k not in {"Ids", "SResults", "RawSResults", "Features", "Labels"}}
    (output_dir / f"{stem}_metrics.json").write_text(
        json.dumps(_json_value(metrics), indent=2, allow_nan=False), encoding="utf-8")
    predictions = {
        "id": results["Ids"],
        "label": np.asarray(results["Labels"]).reshape(-1),
        "prediction": np.asarray(results["SResults"]).reshape(-1),
    }
    if "RawSResults" in results:
        predictions["raw_prediction"] = np.asarray(results["RawSResults"]).reshape(-1)
    pd.DataFrame(predictions).to_csv(output_dir / f"{stem}_predictions.csv", index=False)
    logger.info("Test metrics and predictions saved to %s", output_dir)
    return metrics


def _best_epoch(args, epoch_results):
    minimize = args.KeyEval in ["Loss"]
    best_value = 1e8 if minimize else 0.0
    best_epoch = None
    for epoch, result in enumerate(epoch_results["valid"], start=1):
        value = result[args.KeyEval]
        better = value <= best_value - 1e-6 if minimize else value >= best_value + 1e-6
        if better:
            best_value, best_epoch = value, epoch
    if best_epoch is None:
        raise RuntimeError("No validation checkpoint was selected; inspect validation metrics.")
    return best_epoch


def _evaluate_checkpoint(args, model, dataloader, epoch):
    # A fresh trainer has no calibration left over from another epoch/run.
    trainer = ATIO().getTrain(args)
    if hasattr(trainer, "_current_epoch"):
        trainer._current_epoch = epoch
    # Preserve the original loader's shuffling and batch sizes, while making
    # final evaluation repeatable by the separate --mode test command.
    setup_seed(int(args.seed))
    if args.model_name == "arg_msa":
        trainer.do_test(model, dataloader["valid"], mode="VAL", epoch_num=epoch)
        results = trainer.do_test(model, dataloader["test"], mode="TEST",
                                  return_sample_results=True, epoch_num=epoch)
    else:
        trainer.do_test(model, dataloader["valid"], mode="VAL")
        results = trainer.do_test(model, dataloader["test"], mode="TEST",
                                  return_sample_results=True)
    return _save_evaluation(args, results)


def _run(args, num_workers=0):
    dataloader = MMDataLoader(args, num_workers)
    model = AMIO(args).to(args.device)
    logger.info("Model: %s; trainable parameters: %s", type(model.Model).__name__, count_parameters(model))
    trainer = ATIO().getTrain(args)
    epoch_results = trainer.do_train(model, dataloader, return_epoch_results=True)
    epoch = _best_epoch(args, epoch_results)
    checkpoint = Path(args.model_save_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Validation-selected checkpoint not found: {checkpoint}")
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.to(args.device)
    # Keep the original state_dict checkpoint format. Metadata is a separate,
    # ignored runtime artifact and records the effective configuration/epoch.
    checkpoint.with_suffix(".json").write_text(json.dumps(_json_value({
        "model_name": args.model_name, "dataset_name": args.dataset_name,
        "best_epoch": epoch, "config": dict(args),
    }), indent=2, allow_nan=False), encoding="utf-8")
    _evaluate_checkpoint(args, model, dataloader, epoch)
    del model, dataloader, trainer
    if str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()
    gc.collect()
    return epoch_results


def MSA_run(model_name, dataset_name, config_file=None, config=None, seeds=None,
            custom_feature=None, feature_T=None, feature_A=None, feature_V=None,
            gpu_ids=None, num_workers=None, verbose_level=1,
            model_save_dir="saved_models", res_save_dir="results", log_dir="logs",
            data_root=None):
    model_name, dataset_name = model_name.lower(), dataset_name.lower()
    args = get_config_regression(model_name, dataset_name, config_file or "", data_root=data_root)
    args.update(config or {})
    if args.model_name != model_name or args.dataset_name != dataset_name:
        raise ValueError("Configuration cannot override the selected model or dataset.")
    args.update(train_mode="regression", epochs=int(args.get("epochs", 100)), result_name="",
                custom_feature=custom_feature, feature_T=feature_T, feature_A=feature_A, feature_V=feature_V)
    args.model_save_dir = Path(model_save_dir) / dataset_name
    args.model_save_path = args.model_save_dir / f"{model_name}_{dataset_name}.pth"
    args.res_save_dir = Path(res_save_dir) / model_name
    for directory in (args.model_save_dir, args.res_save_dir, Path(log_dir)):
        directory.mkdir(parents=True, exist_ok=True)
    args.device = assign_gpu([] if gpu_ids is None else list(gpu_ids))
    if str(args.device).startswith("cuda"):
        torch.cuda.set_device(args.device)
    run_workers = int(args.get("num_workers", 0) if num_workers is None else num_workers)
    _set_logger(log_dir, model_name, dataset_name, verbose_level)
    try:
        for i, seed in enumerate(seeds or [1111]):
            setup_seed(seed)
            args.seed, args.cur_seed = seed, i + 1
            logger.info("Training %s on %s; seed=%s, epochs=%s", model_name, dataset_name, seed, args.epochs)
            epoch_results = _run(args, run_workers)
        rows = _build_summary_rows(epoch_results)
        if len(rows) != args.epochs:
            raise RuntimeError(f"Summary rows {len(rows)} do not match epochs {args.epochs}.")
        csv_file = args.res_save_dir / f"{model_name}_{dataset_name}.csv"
        pd.DataFrame(rows, columns=SUMMARY_COLUMNS).to_csv(csv_file, index=False, float_format="%.6f")
        _audit_csv_file(csv_file, len(rows))
        return epoch_results
    finally:
        _close_logger_handlers()


def _build_runtime_config(cli_args):
    keys = ("epochs", "batch_size", "num_workers", "learning_rate", "weight_decay", "show_progress")
    config = {key: getattr(cli_args, key) for key in keys if getattr(cli_args, key, None) is not None}
    if "learning_rate" in config:
        config["lr"] = config["learning_rate"]
    return config


def train(cli_args):
    return MSA_run(
        model_name=cli_args.model_name, dataset_name=cli_args.dataset_name,
        config_file=cli_args.config_file, config=_build_runtime_config(cli_args), seeds=[cli_args.seed],
        custom_feature=cli_args.custom_feature, feature_T=cli_args.feature_T,
        feature_A=cli_args.feature_A, feature_V=cli_args.feature_V,
        gpu_ids=cli_args.gpu_ids, num_workers=cli_args.num_workers, verbose_level=cli_args.verbose_level,
        model_save_dir=cli_args.model_save_dir, res_save_dir=cli_args.res_save_dir,
        log_dir=cli_args.log_dir, data_root=cli_args.data_root,
    )


def test(cli_args):
    checkpoint = Path(cli_args.checkpoint) if cli_args.checkpoint else None
    if checkpoint is None or not checkpoint.is_file():
        raise FileNotFoundError(f"--mode test requires an existing --checkpoint: {checkpoint}")
    args = get_config_regression(cli_args.model_name, cli_args.dataset_name,
                                 cli_args.config_file or "", data_root=cli_args.data_root)
    metadata_path = checkpoint.with_suffix(".json")
    epoch = cli_args.checkpoint_epoch
    saved = {}
    saved_features = {}
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (metadata["model_name"], metadata["dataset_name"]) != (args.model_name, args.dataset_name):
            raise ValueError("Checkpoint metadata disagrees with the selected model/dataset.")
        runtime_keys = {"featurePath", "device", "custom_feature", "feature_T", "feature_A", "feature_V",
                        "model_save_dir", "model_save_path", "res_save_dir", "result_name", "num_workers"}
        saved = {k: v for k, v in metadata["config"].items() if k not in runtime_keys}
        saved_features = {key: metadata["config"].get(key)
                          for key in ("custom_feature", "feature_T", "feature_A", "feature_V")}
        args.update(saved)
        if epoch is None:
            epoch = int(metadata["best_epoch"])
    if epoch is None:
        # Existing checkpoints contain weights only; a caller must supply the
        # validation-selected epoch for epoch-dependent calibration settings.
        raise ValueError("Legacy checkpoint has no metadata; provide --checkpoint_epoch with its best validation epoch.")
    if epoch <= 0:
        raise ValueError("Checkpoint epoch must be positive.")
    args.update(_build_runtime_config(cli_args))
    args.update(train_mode="regression", seed=saved.get("seed", cli_args.seed), cur_seed=1)
    for key in ("custom_feature", "feature_T", "feature_A", "feature_V"):
        supplied = getattr(cli_args, key)
        args[key] = supplied if supplied is not None else saved_features.get(key)
        if args[key] and not Path(args[key]).is_file():
            raise FileNotFoundError(f"Recorded {key} is unavailable; provide --{key} for its new location: {args[key]}")
    args.device = assign_gpu([] if cli_args.gpu_ids is None else list(cli_args.gpu_ids))
    args.res_save_dir = Path(cli_args.res_save_dir) / args.model_name
    Path(cli_args.log_dir).mkdir(parents=True, exist_ok=True)
    _set_logger(cli_args.log_dir, args.model_name, args.dataset_name, cli_args.verbose_level)
    try:
        setup_seed(int(args.seed))
        dataloader = MMDataLoader(args, cli_args.num_workers or 0)
        model = AMIO(args).to(args.device)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        model.to(args.device)
        return _evaluate_checkpoint(args, model, dataloader, epoch)
    finally:
        _close_logger_handlers()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "test"), default="train")
    parser.add_argument("--model_name", choices=SUPPORTED_MODELS, default="arg_msa")
    parser.add_argument("--dataset_name", choices=SUPPORTED_DATASETS, default="sims")
    parser.add_argument("--config_file", default=None)
    parser.add_argument("--data_root", default=None, help="Dataset root; overrides MSA_DATA_ROOT.")
    parser.add_argument("--custom_feature", default=None)
    parser.add_argument("--feature_T", default=None)
    parser.add_argument("--feature_A", default=None)
    parser.add_argument("--feature_V", default=None)
    parser.add_argument("--gpu_ids", type=int, nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--learning_rate", "--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--show_progress", choices=("true", "false"), default=None)
    parser.add_argument("--verbose_level", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--model_save_dir", default="saved_models")
    parser.add_argument("--res_save_dir", default="results")
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--checkpoint", default=None, help="Validation-selected .pth checkpoint for test mode.")
    parser.add_argument("--checkpoint_epoch", type=int, default=None,
                        help="Best validation epoch, required for legacy checkpoints without a JSON sidecar.")
    cli_args = parser.parse_args()
    if cli_args.show_progress is not None:
        cli_args.show_progress = cli_args.show_progress == "true"
    if cli_args.epochs is not None and cli_args.epochs <= 0:
        parser.error("--epochs must be positive")
    if cli_args.batch_size is not None and cli_args.batch_size <= 0:
        parser.error("--batch_size must be positive")
    if cli_args.mode == "test" and not cli_args.checkpoint:
        parser.error("--mode test requires --checkpoint")
    train(cli_args) if cli_args.mode == "train" else test(cli_args)
