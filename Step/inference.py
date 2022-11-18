import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
import Instances.instance as instance
import Utils.labels_ids as labels_ids
import torch.nn.functional as F
import os


def inference(args, conf):

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=conf.train.max_epoch,
        log_every_n_steps=1,
    )

    dataloader, model, args, conf = instance.load_instance(args, conf)

    trainer.test(model=model, datamodule=dataloader)
    predictions = trainer.predict(
        model=model,
        datamodule=dataloader,
    )

    # 저번 베이스라인 코드와 이번 베이스라인 코드 결합

    predictions = list(i for i in torch.cat(predictions))  # logits들의 배열

    output_pred = [np.argmax(logit, axis=-1).item() for logit in predictions]
    output_prob = [F.softmax(logit, dim=-1).tolist() for logit in predictions]
    pred_answer = labels_ids.num_to_label(output_pred)

    predict_data = pd.read_csv(conf.path.predict_path)
    predict_id = predict_data["id"]
    output = pd.DataFrame(
        {
            "id": predict_id,
            "pred_label": pred_answer,
            "probs": output_prob,
        }
    )

    # 만약 prediction 폴더가 없다면 생성합니다
    dir_path = "./prediction"
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    # print(args.saved_model)  ex) SaveModels/klue/roberta-base_usual-dust-1/epoch=0-val_micro_f1=76.8685077273879.ckpt 인 경우
    # usual-dust-1_epoch=0 로 들어감

    key = "_".join([args.saved_model.split("/")[-2].split("_")[1], args.saved_model.split("/")[-1].split("-")[0]])
    output.to_csv(f"./prediction/submission_{key}.csv", index=False)  # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
