from pytorch_lightning.loggers import WandbLogger

import pytorch_lightning as pl
import Instances.instance as instance
import wandb

import Utils.utils as utils


def train(args, conf):

    project_name = conf.wandb.project
    dataloader, model = instance.new_instance(conf)
    wandb_logger = WandbLogger(project=project_name)

    save_path = (
        f"{conf.path.save_path}{conf.model.model_name}_{wandb_logger.experiment.name}/"
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=conf.train.max_epoch,
        log_every_n_steps=1,
        logger=wandb_logger,
        callbacks=[
            utils.early_stop(
                monitor=utils.monitor_dict[conf.utils.early_stop_monitor]["monitor"],
                mode=utils.monitor_dict[conf.utils.early_stop_monitor]["mode"],
                patience=conf.utils.patience,
            ),
            utils.best_save(
                save_path=save_path,
                top_k=conf.utils.top_k,
                monitor=utils.monitor_dict[conf.utils.best_save_monitor]["monitor"],
                mode=utils.monitor_dict[conf.utils.best_save_monitor]["mode"],
                filename="{epoch}-{val_micro_f1}",
            ),
        ],
    )
    trainer.fit(model=model, datamodule=dataloader)
    test_micro_f1 = trainer.test(model=model, datamodule=dataloader)
    wandb.finish()

    # 마지막 모델을 저장합니다
    test_micro_f1 = test_micro_f1[0]["test_micro_f1"]
    trainer.save_checkpoint(
        f"{save_path}epoch={conf.train.max_epoch-1}-test_micro_f1={test_micro_f1}.ckpt"
    )
