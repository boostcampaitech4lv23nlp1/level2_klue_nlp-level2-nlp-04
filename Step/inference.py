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

    key = "_".join([args.saved_model.split("/")[-2].split("_")[1], args.saved_model.split("/")[-1].split("-")[0]])
    output.to_csv(f"./prediction/submission_{key}.csv", index=False)  # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.


def k_fold_inference(args, conf):
    test_list = []
    predictions_output_prob_list = []  # softmax를 취한 값 즉 확률을 전부 리스트에 넣어줍니다
    for k in range(conf.k_fold.num_folds):
        trainer = pl.Trainer(gpus=1, max_epochs=conf.train.max_epoch, log_every_n_steps=1)

        k_dataloader, k_model = instance.kfold_load_instance(args, conf, k)

        test_micro_f1 = trainer.test(model=k_model, datamodule=k_dataloader)
        test_micro_f1 = test_micro_f1[0]["test_micro_f1"]

        test_list.append(test_micro_f1)

        predictions = trainer.predict(
            model=k_model,
            datamodule=k_dataloader,
        )

        predictions = list(i for i in torch.cat(predictions))  # logits들의 배열
        output_prob = [F.softmax(logit, dim=-1).numpy() for logit in predictions]
        predictions_output_prob_list.append(output_prob)

    average_prob = np.stack(predictions_output_prob_list, axis=1).mean(axis=1)
    average_pred = [np.argmax(logit, axis=-1).item() for logit in average_prob]
    pred_answer = labels_ids.num_to_label(average_pred)

    average_prob = [list(i) for i in average_prob]  # 리스트 형태로 변환

    predict_data = pd.read_csv(conf.path.predict_path)
    predict_id = predict_data["id"]
    output = pd.DataFrame(
        {
            "id": predict_id,
            "pred_label": pred_answer,
            "probs": average_prob,
        }
    )

    # 만약 prediction 폴더가 없다면 생성합니다
    dir_path = "./prediction"
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    key = args.saved_model.split("/")[-1]
    output.to_csv(f"./prediction/submission_{key}.csv", index=False)  # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
