import torch
import os
import seaborn as sns
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import torch.nn.functional as F
import pickle

from sklearn.metrics import confusion_matrix

from Instances import instance



def get_predictions(args, conf):
    """지정된 모델을 불러와서 test set에 대한 prediction을 반환

    Args:
        args (_type_): args
        conf (_type_): conf
    
    Returns:
        Tuple(torch.Tensor): test set에 대한 prediction, 전체 class에 대한 logit값
    """

    with open('./dict_label_to_num.pkl', 'rb') as f:
        label_to_num = pickle.load(f)

    test = pd.read_csv('../dataset/train/test.csv')
    y_true = list(test['label'].apply(lambda x: label_to_num[x]))

    test['label'] = test['label'].apply(lambda x: 100)
    test.to_csv('../dataset/test/test_for_pred.csv', index=False)
    


    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=conf.train.max_epoch,
        log_every_n_steps=1,
    )

    dataloader, model, args, conf = instance.load_instance(args, conf)

    dataloader.predict_path = '../dataset/test/test_for_pred.csv'

    # trainer.test(model=model, datamodule=dataloader)
    predictions = trainer.predict(model=model, datamodule=dataloader)

    predictions = list(i for i in torch.cat(predictions))

    y_pred = [np.argmax(logit, axis=-1).item() for logit in predictions]
    y_prob = [F.softmax(logit, dim=-1).tolist() for logit in predictions]

    return y_true, y_pred, y_prob


def plot_confusion_matrix(args, conf):
    """저장된 모델을 불러와서 test set에 대한 confusion matrix를 구하여
        plot 한 뒤 저장하는 함수

    Args:
        args (_type_): args
        conf (_type_): conf

    Returns:
        None: .png 확장자로 './analysis/cm_{model_name}.png' 으로 저장
    """

    y_true, y_pred, _ = get_predictions(args, conf)

    dir_path = "./analysis"
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    key = "_".join([args.saved_model.split("/")[-2].split("_")[1], args.saved_model.split("/")[-1].split("-")[0]])
    file_name = 'cm_' + key + '.png'


    confmat = confusion_matrix(y_true, y_pred, normalize='true')
    fig = sns.heatmap(confmat).get_figure()

    fig.savefig(f'{dir_path}/{file_name}')

    return 0


def error_analysis(args, conf): # 아직!!
    """저장된 모델을 불러와서 test set에 대한 error analysis를 진행한 후
        csv형태로 저장하는 함수

    Args:
        args (_type_): args
        conf (_type_): conf

    Returns:
        None: .csv 확장자로 './analysis/error_{model_name}.csv'로 저장
    """
    """
    y_true, y_pred, y_prob = get_predictions(args, conf)

    dir_path = "./analysis"
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    key = "_".join([args.saved_model.split("/")[-2].split("_")[1], args.saved_model.split("/")[-1].split("-")[0]])
    file_name = 'error_' + key + '.csv'

    max_sample = 100

    ### TODO: max_sample개만큼 로스가 큰 순서대로 파일에 저장

    ###
    """
    return 0

