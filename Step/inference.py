import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
import Instances.instance as instance


def inference(args, conf):

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=conf.train.max_epoch,
        log_every_n_steps=1,
    )

    dataloader, model, args, conf = instance.load_instance(args, conf)

    test_micro_f1 = trainer.test(model=model, datamodule=dataloader)
    test_micro_f1 = test_micro_f1[0]["test_micro_f1"]
