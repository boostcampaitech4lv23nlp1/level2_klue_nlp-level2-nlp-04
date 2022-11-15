from tqdm.auto import tqdm
from torch.optim.lr_scheduler import StepLR, ExponentialLR, LambdaLR


import transformers
import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
import Utils.utils as utils
import Utils.metric as metric
from . import lr_scheduler_Func


class Model(pl.LightningModule):
    def __init__(self, conf, new_vocab_size):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = conf.model.model_name
        self.lr = conf.train.lr
        self.model_config = transformers.AutoConfig.from_pretrained(self.model_name)
        self.model_config.num_labels = 30
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            self.model_name, config=self.model_config
        )

        ## print문 주석 해제해서 임베딩 차원이 어떻게 바뀌는지 한번 확인해보세요
        # print(self.plm)

        self.plm.resize_token_embeddings(new_vocab_size)  # vocab 사이즈 조정 (새로운 토큰 추가에 의함)

        # token_type_embeddings을 위한 공간 (만약 1차원이라면((token_type_embeddings): Embedding(1, 768)))
        # SequenceClassification 이어서 한번 감싸져서 나옴 -> 이름이 중요해짐
        if self.plm.config.type_vocab_size == 1:
            self.plm.config.type_vocab_size = 2
            if type(self.plm).__name__ == "BertForSequenceClassification":  # bert 부분
                single_emb = self.plm.roberta.embeddings.token_type_embeddings
                self.plm.bert.embeddings.token_type_embeddings = torch.nn.Embedding(
                    2, single_emb.embedding_dim
                )
                self.plm.bert.embeddings.token_type_embeddings.weight = (
                    torch.nn.Parameter(single_emb.weight.repeat([2, 1]))
                )

            elif (
                type(self.plm).__name__ == "RobertaForSequenceClassification"
            ):  # roberta 부분
                single_emb = self.plm.roberta.embeddings.token_type_embeddings
                self.plm.roberta.embeddings.token_type_embeddings = torch.nn.Embedding(
                    2, single_emb.embedding_dim
                )
                self.plm.roberta.embeddings.token_type_embeddings.weight = (
                    torch.nn.Parameter(single_emb.weight.repeat([2, 1]))
                )
            else:
                print("model을 추가해주세요")
                exit(1)

        # print(self.plm)

        self.loss_func = utils.loss_dict[conf.train.loss]
        self.use_freeze = conf.train.use_freeze

        if self.use_freeze:
            self.freeze()

    def forward(self, items):  ## **items
        x = self.plm(
            input_ids=items["input_ids"],
            attention_mask=items["attention_mask"],
            token_type_ids=items["token_type_ids"],
        )[
            "logits"
        ]  # cls -> classifier 한 결과를 뱉음
        return x

    def training_step(self, batch, batch_idx):
        items = batch

        logits = self(items)
        loss = self.loss_func(logits, items["labels"].long())
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        items = batch

        logits = self(items)
        loss = self.loss_func(logits, items["labels"].long())

        self.log("val_loss", loss)
        self.log(
            "val_micro_f1",
            metric.klue_re_micro_f1(
                logits.argmax(-1).cpu().numpy(), items["labels"].cpu().numpy()
            ),
        )  # f1 score를 계산합니다
        # TODO: auprc 구현 log 할 수 있도록 하여야함
        # TODO: 마지막에 계산된 값만 저장됨, 평균을 취하거나 한번에 기록될 수 있게 조치를 취햐아함
        return loss

    def test_step(self, batch, batch_idx):
        items = batch
        logits = self(items)

        self.log(
            "test_micro_f1",
            metric.klue_re_micro_f1(
                logits.argmax(-1).cpu().numpy(), items["labels"].cpu().numpy()
            ),
        )  # f1 score를 계산합니다

    def predict_step(self, batch, batch_idx):
        items = batch
        logits = self(items)

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def freeze(self):
        for name, param in self.plm.named_parameters():
            freeze_list = []
            if name in freeze_list:
                param.requires_grad = False
