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

# TODO: auprc log 할 수 있도록 하여야 함


class Model(pl.LightningModule):
    def __init__(self, conf, new_vocab_size):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = conf.model.model_name
        self.lr = conf.train.lr
        self.model_config = transformers.AutoConfig.from_pretrained(self.model_name)
        self.model_config.num_labels = 30
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(self.model_name, config=self.model_config)
        self.warm_up = conf.train.warm_up
        ## print문 주석 해제해서 임베딩 차원이 어떻게 바뀌는지 한번 확인해보세요
        # print(self.plm)

        self.plm.resize_token_embeddings(new_vocab_size)  # vocab 사이즈 조정 (새로운 토큰 추가에 의함)

        # token_type_embeddings을 위한 공간 (만약 1차원이라면((token_type_embeddings): Embedding(1, 768)))
        # SequenceClassification 이어서 한번 감싸져서 나옴 -> 이름이 중요해짐
        if self.plm.config.type_vocab_size == 1:
            self.plm.config.type_vocab_size = 2
            if type(self.plm).__name__ == "BertForSequenceClassification":  # bert 부분
                single_emb = self.plm.bert.embeddings.token_type_embeddings
                self.plm.bert.embeddings.token_type_embeddings = torch.nn.Embedding(2, single_emb.embedding_dim)
                self.plm.bert.embeddings.token_type_embeddings.weight = torch.nn.Parameter(single_emb.weight.repeat([2, 1]))

            elif type(self.plm).__name__ == "RobertaForSequenceClassification":  # roberta 부분
                single_emb = self.plm.roberta.embeddings.token_type_embeddings
                self.plm.roberta.embeddings.token_type_embeddings = torch.nn.Embedding(2, single_emb.embedding_dim)
                self.plm.roberta.embeddings.token_type_embeddings.weight = torch.nn.Parameter(single_emb.weight.repeat([2, 1]))

            elif type(self.plm).__name__ == "ElectraForSequenceClassification":  # electra 부분
                single_emb = self.plm.electra.embeddings.token_type_embeddings
                self.plm.electra.embeddings.token_type_embeddings = torch.nn.Embedding(2, single_emb.embedding_dim)
                self.plm.electra.embeddings.token_type_embeddings.weight = torch.nn.Parameter(single_emb.weight.repeat([2, 1]))

            elif type(self.plm).__name__ == "BigBirdForSequenceClassification":  # bigbird 부분
                single_emb = self.plm.bert.embeddings.token_type_embeddings
                self.plm.bert.embeddings.token_type_embeddings = torch.nn.Embedding(2, single_emb.embedding_dim)
                self.plm.bert.embeddings.token_type_embeddings.weight = torch.nn.Parameter(single_emb.weight.repeat([2, 1]))

            else:
                print("model을 추가해주세요")
                exit(1)

        # print(self.plm)

        self.loss_func = utils.loss_dict[conf.train.loss]
        self.use_freeze = conf.train.use_freeze

        if self.use_freeze:
            self.freeze()

    def forward(self, items):  ## **items
        x = self.plm(input_ids=items["input_ids"], attention_mask=items["attention_mask"], token_type_ids=items["token_type_ids"],)[
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
        pred = logits.argmax(-1)  # pred 한 라벨

        return {"val_loss": loss, "pred": pred, "label": items["labels"]}

    def validation_epoch_end(self, outputs):
        pred_all = torch.concat([x["pred"] for x in outputs])  # 배치당 예측한 라벨들을 전부 concat 하여 전체 예측 텐서 생성
        label_all = torch.concat([x["label"] for x in outputs])  # 배치당 정답 라벨을 전부 concat하여 전체 정답 텐서 생성
        val_f1 = metric.klue_re_micro_f1(pred_all.cpu().numpy(), label_all.cpu().numpy())  # micro_f1 계산
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()  # 배치당 loss를 전부 stack으로 받아 평균 취함

        self.log("val_micro_f1", val_f1)
        self.log("val_loss", avg_loss)

    def test_step(self, batch, batch_idx):
        items = batch

        logits = self(items)
        pred = logits.argmax(-1)  # pred 한 라벨
        return {"pred": pred, "label": items["labels"]}

    def test_epoch_end(self, outputs):
        pred_all = torch.concat([x["pred"] for x in outputs])  # 배치당 예측한 라벨들을 전부 concat 하여 전체 예측 텐서 생성
        label_all = torch.concat([x["label"] for x in outputs])  # 배치당 정답 라벨을 전부 concat하여 전체 정답 텐서 생성
        test_f1 = metric.klue_re_micro_f1(pred_all.cpu().numpy(), label_all.cpu().numpy())  # micro_f1 계산

        self.log("test_micro_f1", test_f1)

    def predict_step(self, batch, batch_idx):
        items = batch
        logits = self(items)

        return logits.squeeze()

    ## Warm-up 단계 밑의 링크 참고하여 작성
    ## https://pytorch-lightning.readthedocs.io/en/stable/common/optimization.html
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)  # 추후에 알려드릴 예정
        scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lambda step: min(1.0, float(step + 1) / (self.warm_up + 1)))
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def freeze(self):
        for name, param in self.plm.named_parameters():
            freeze_list = []
            if name in freeze_list:
                param.requires_grad = False
