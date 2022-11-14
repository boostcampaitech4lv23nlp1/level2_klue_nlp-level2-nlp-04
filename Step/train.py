from pytorch_lightning.loggers import WandbLogger

import pytorch_lightning as pl
import Instances.instance as instance
import wandb

import Utils.utils as utils


def train(args, conf):

    project_name = conf.wandb.project
    dataloader, model = instance.new_instance(conf)
    wandb_logger = WandbLogger(project=project_name)

    save_path = f"{conf.path.save_path}{conf.model.model_name}_{wandb_logger.experiment.name}/"

    trainer = pl.Trainer(
        accelerator='gpu', devices=1,
        max_epochs=conf.train.max_epoch,
        log_every_n_steps=1,
        logger=wandb_logger,
    )
    trainer.fit(model=model, datamodule=dataloader)
    wandb.finish()
