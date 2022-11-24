import os
import Instances.Models.loss as loss
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def early_stop(monitor, patience, mode):
    early_stop_callback = EarlyStopping(monitor=monitor, min_delta=0.00, patience=patience, verbose=False, mode=mode)
    return early_stop_callback


def best_save(save_path, top_k, monitor, mode, filename):
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_path,
        save_top_k=top_k,
        monitor=monitor,
        mode=mode,
        filename=filename,
    )
    return checkpoint_callback


loss_dict = {"l1": loss.L1_loss, "mse": loss.MSE_loss, "bce": loss.BCEWithLogitsLoss, "rmse": loss.RMSE_loss, "huber": loss.HUBER_loss, "ce": loss.CrossEntropyLoss, "focal": loss.Focal_loss}


monitor_dict = {
    "val_loss": {"monitor": "val_loss", "mode": "min"},
    "val_micro_f1": {"monitor": "val_micro_f1", "mode": "max"},
}
