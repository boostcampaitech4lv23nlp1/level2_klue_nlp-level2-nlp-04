from tqdm.auto import tqdm
from torch.optim.lr_scheduler import StepLR, ExponentialLR, LambdaLR


import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(self.model_name, config=self.model_config)
        self.warm_up = conf.train.warm_up
        ## print문 주석 해제해서 임베딩 차원이 어떻게 바뀌는지 한번 확인해보세요
        # print(self.plm)

        self.plm.resize_token_embeddings(new_vocab_size)  # vocab 사이즈 조정 (새로운 토큰 추가에 의함)

        # token_type_embeddings을 위한 공간 (만약 1차원이라면((token_type_embeddings): Embedding(1, 768)))

        if self.plm.config.type_vocab_size == 1:  # base_model을 통해 일관된 모양으로 받을 수 있습니다 따라서 모두 통일된 형태입니다
            self.plm.config.type_vocab_size = 2
            single_emb = self.plm.base_model.embeddings.token_type_embeddings
            self.plm.base_model.embeddings.token_type_embeddings = torch.nn.Embedding(2, single_emb.embedding_dim)
            self.plm.base_model.embeddings.token_type_embeddings.weight = torch.nn.Parameter(single_emb.weight.repeat([2, 1]))

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
        prob = F.softmax(logits, dim=-1)  # 라벨 전체

        return {"val_loss": loss, "pred": pred, "prob": prob, "label": items["labels"]}

    def validation_epoch_end(self, outputs):
        pred_all = torch.concat([x["pred"] for x in outputs])  # 배치당 예측한 라벨들을 전부 concat 하여 전체 예측 텐서 생성
        prob_all = torch.concat([x["prob"] for x in outputs])
        label_all = torch.concat([x["label"] for x in outputs])  # 배치당 정답 라벨을 전부 concat하여 전체 정답 텐서 생성
        val_f1 = metric.klue_re_micro_f1(pred_all.cpu().numpy(), label_all.cpu().numpy())  # micro_f1 계산
        val_auprc = metric.klue_re_auprc(prob_all.cpu().numpy(), label_all.cpu().numpy())
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()  # 배치당 loss를 전부 stack으로 받아 평균 취함

        self.log("val_micro_f1", val_f1)
        self.log("val_auprc", val_auprc)
        self.log("val_loss", avg_loss)

    def test_step(self, batch, batch_idx):
        items = batch

        logits = self(items)
        pred = logits.argmax(-1)  # pred 한 라벨
        prob = F.softmax(logits, dim=-1)  # 라벨 전체
        return {"pred": pred, "prob": prob, "label": items["labels"]}

    def test_epoch_end(self, outputs):
        pred_all = torch.concat([x["pred"] for x in outputs])  # 배치당 예측한 라벨들을 전부 concat 하여 전체 예측 텐서 생성
        prob_all = torch.concat([x["prob"] for x in outputs])
        label_all = torch.concat([x["label"] for x in outputs])  # 배치당 정답 라벨을 전부 concat하여 전체 정답 텐서 생성
        test_f1 = metric.klue_re_micro_f1(pred_all.cpu().numpy(), label_all.cpu().numpy())  # micro_f1 계산
        test_auprc = metric.klue_re_auprc(prob_all.cpu().numpy(), label_all.cpu().numpy())

        self.log("test_micro_f1", test_f1)
        self.log("test_auprc", test_auprc)

    def predict_step(self, batch, batch_idx):
        items = batch
        logits = self(items)

        return logits.squeeze()

    ## Warm-up 단계 밑의 링크 참고하여 작성
    ## https://pytorch-lightning.readthedocs.io/en/stable/common/optimization.html
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)  # 추후에 알려드릴 예정
        scheduler = LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda step: min(1.0, float(step + 1) / (self.warm_up + 1)),
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def freeze(self):
        for name, param in self.plm.named_parameters():
            freeze_list = []
            if name in freeze_list:
                param.requires_grad = False


class ExampleModel1(pl.LightningModule):
    def __init__(self, conf, new_vocab_size):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = conf.model.model_name
        self.lr = conf.train.lr
        self.model_config = transformers.AutoConfig.from_pretrained(self.model_name)  # classifier의 input 차원을 얻어오기 위해 모델 정보를 불러옵니다
        self.input_dim = self.model_config.hidden_size  # input 차원입니다(cls토큰의 차원)
        self.num_labels = 30  # 최종 output label의 개수입니다(차원)
        # self.input_dim = self.model_config.d_model  # 가끔 다른 모델 구조는 input 차원을 d_model로 사용하기도 합니다(cls토큰의 차원)

        self.plm = transformers.AutoModel.from_pretrained(self.model_name)
        self.warm_up = conf.train.warm_up

        ## print문 주석 해제해서 임베딩 차원이 어떻게 바뀌는지 한번 확인해보세요
        # print(self.plm)

        self.plm.resize_token_embeddings(new_vocab_size)  # vocab 사이즈 조정 (새로운 토큰 추가에 의함)

        if self.plm.config.type_vocab_size == 1:  # classifier 모듈이 없어, 감싸져있지 않습니다 따라서 모두 통일된 형태입니다
            self.plm.config.type_vocab_size = 2
            single_emb = self.plm.embeddings.token_type_embeddings
            self.plm.embeddings.token_type_embeddings = torch.nn.Embedding(2, single_emb.embedding_dim)
            self.plm.embeddings.token_type_embeddings.weight = torch.nn.Parameter(single_emb.weight.repeat([2, 1]))

        # print(self.plm)

        self.loss_func = utils.loss_dict[conf.train.loss]
        self.use_freeze = conf.train.use_freeze

        if self.use_freeze:
            self.freeze()

        # classifier를 정의해줍니다, 최종 num_label은 30으로 고정되어야합니다.
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.input_dim, self.num_labels),
        )

    def forward(self, items):  ## **items
        x = self.plm(input_ids=items["input_ids"], attention_mask=items["attention_mask"], token_type_ids=items["token_type_ids"],)[
            0
        ]  # pooler까지 거친 최종적인 output입니다
        x = x[:, 0, :]
        x = self.classifier(x)  # 분류기를 거칩니다
        return x  # 분류기를 거친 최종 30차원 탠서를 반환해줍니다

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
        prob = F.softmax(logits, dim=-1)  # 라벨 전체

        return {"val_loss": loss, "pred": pred, "prob": prob, "label": items["labels"]}

    def validation_epoch_end(self, outputs):
        pred_all = torch.concat([x["pred"] for x in outputs])  # 배치당 예측한 라벨들을 전부 concat 하여 전체 예측 텐서 생성
        prob_all = torch.concat([x["prob"] for x in outputs])
        label_all = torch.concat([x["label"] for x in outputs])  # 배치당 정답 라벨을 전부 concat하여 전체 정답 텐서 생성
        val_f1 = metric.klue_re_micro_f1(pred_all.cpu().numpy(), label_all.cpu().numpy())  # micro_f1 계산
        val_auprc = metric.klue_re_auprc(prob_all.cpu().numpy(), label_all.cpu().numpy())
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()  # 배치당 loss를 전부 stack으로 받아 평균 취함

        self.log("val_micro_f1", val_f1)
        self.log("val_auprc", val_auprc)
        self.log("val_loss", avg_loss)

    def test_step(self, batch, batch_idx):
        items = batch

        logits = self(items)
        pred = logits.argmax(-1)  # pred 한 라벨
        prob = F.softmax(logits, dim=-1)  # 라벨 전체
        return {"pred": pred, "prob": prob, "label": items["labels"]}

    def test_epoch_end(self, outputs):
        pred_all = torch.concat([x["pred"] for x in outputs])  # 배치당 예측한 라벨들을 전부 concat 하여 전체 예측 텐서 생성
        prob_all = torch.concat([x["prob"] for x in outputs])
        label_all = torch.concat([x["label"] for x in outputs])  # 배치당 정답 라벨을 전부 concat하여 전체 정답 텐서 생성
        test_f1 = metric.klue_re_micro_f1(pred_all.cpu().numpy(), label_all.cpu().numpy())  # micro_f1 계산
        test_auprc = metric.klue_re_auprc(prob_all.cpu().numpy(), label_all.cpu().numpy())

        self.log("test_micro_f1", test_f1)
        self.log("test_auprc", test_auprc)

    def predict_step(self, batch, batch_idx):
        items = batch
        logits = self(items)

        return logits.squeeze()

    ## Warm-up 단계 밑의 링크 참고하여 작성
    ## https://pytorch-lightning.readthedocs.io/en/stable/common/optimization.html
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)  # 추후에 알려드릴 예정
        scheduler = LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda step: min(1.0, float(step + 1) / (self.warm_up + 1)),
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def freeze(self):
        for name, param in self.plm.named_parameters():
            freeze_list = []
            if name in freeze_list:
                param.requires_grad = False


# 상속을 받으면 기존에 구현되어 있는 코드를 재활용할 수 있습니다
# 상속받은 자식클래스에 classifier가 없다면 부모클래스에 있던 classifier가 print 할때는 나오지만 forward 과정에서는 사용되지 않습니다
# 이렇게 안하시고 복붙하셔도 상관없습니다
class ExampleModel2(ExampleModel1):
    def __init__(self, conf, new_vocab_size):
        super().__init__(conf, new_vocab_size)

        self.hidden_dim = 1024

        self.classifier = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim), nn.ReLU(), nn.Dropout(0.2))
        self.classifier2 = nn.Sequential(nn.Linear(self.input_dim + self.hidden_dim, self.num_labels))

    def forward(self, items):  ## **items
        x = self.plm(input_ids=items["input_ids"], attention_mask=items["attention_mask"], token_type_ids=items["token_type_ids"],)[
            0
        ]  # 최종적인 output입니다
        x = x[:, 0, :]  # 배치의, 0번째 인덱스(CLS)의 모든 요소(768)을 가져옵니다
        y = self.classifier(x)
        x = torch.cat((x, y), dim=1)
        x = self.classifier2(x)
        return x


class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.0, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)


class RBERT(pl.LightningModule):
    def __init__(self, conf, new_vocab_size):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = conf.model.model_name
        self.lr = conf.train.lr
        self.model_config = transformers.AutoConfig.from_pretrained(self.model_name)
        self.model_config.num_labels = 30
        self.plm = transformers.AutoModel.from_pretrained(self.model_name, config=self.model_config)
        self.warm_up = conf.train.warm_up
        self.dropout_rate = 0.1

        self.plm.resize_token_embeddings(new_vocab_size)

        if self.plm.config.type_vocab_size == 1:
            self.plm.config.type_vocab_size = 2
            single_emb = self.plm.base_model.embeddings.token_type_embeddings
            self.plm.base_model.embeddings.token_type_embeddings = torch.nn.Embedding(2, single_emb.embedding_dim)
            self.plm.base_model.embeddings.token_type_embeddings.weight = torch.nn.Parameter(single_emb.weight.repeat([2, 1]))

        self.cls_fc_layer = FCLayer(self.model_config.hidden_size, self.model_config.hidden_size, self.dropout_rate)
        self.entity_fc_layer = FCLayer(self.model_config.hidden_size, self.model_config.hidden_size, self.dropout_rate)
        self.label_classifier = FCLayer(
            self.model_config.hidden_size * 3,
            self.model_config.num_labels,
            self.dropout_rate,
            use_activation=False,
        )

        self.loss_func = utils.loss_dict[conf.train.loss]
        self.use_freeze = conf.train.use_freeze

        if self.use_freeze:
            self.freeze()

    def entity_average(hidden_output, e_mask):
        e_mask_unsqueeze = e_mask.unsqueeze(1)
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)

        sum_vector = torch.bmm(
            e_mask_unsqueeze.float(),
            hidden_output.float(),
        ).squeeze(1)
        avg_vetor = sum_vector.float() / length_tensor.float()
        return avg_vetor

    def forward(self, items):
        outputs = self.plm(
            input_ids=items["input_ids"],
            attention_mask=items["attention_mask"],
            token_type_ids=items["token_type_ids"],
        )
        sequence_output = outputs.last_hidden_state
        pooler_output = outputs.pooler_output

        # Average
        e1_h = self.entity_average(sequence_output, items["e1_mask"])
        e2_h = self.entity_average(sequence_output, items["e2_mask"])

        # Dropout -> tanh -> fc_layer (Share FC layer for e1 and e2)
        pooler_output = self.cls_fc_layer(pooler_output)
        e1_h = self.entity_fc_layer(e1_h)
        e2_h = self.entity_fc_layer(e2_h)

        # Concat -> fc_layer
        concat_h = torch.cat([pooler_output, e1_h, e2_h], dim=-1)
        logits = self.label_classifier(concat_h)
        return logits

    def training_step(self, batch, batch_idx):
        items = batch
        logits = self(items)
        loss = self.loss_func(logits.view(-1, self.model_config.num_labels), items["labels"].view(-1))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        items = batch
        logits = self(items)
        loss = self.loss_func(logits.view(-1, self.model_config.num_labels), items["labels"].view(-1))
        pred = logits.argmax(-1)

        return {"val_loss": loss, "pred": pred, "label": items["labels"]}

    def validation_epoch_end(self, outputs):
        pred_all = torch.concat([x["pred"] for x in outputs])
        label_all = torch.concat([x["label"] for x in outputs])
        val_f1 = metric.klue_re_micro_f1(pred_all.cpu().numpy(), label_all.cpu().numpy())  # micro_f1 계산
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("val_micro_f1", val_f1)
        self.log("val_loss", avg_loss)

    def test_step(self, batch, batch_idx):
        items = batch
        logits = self(items)
        pred = logits.argmax(-1)  # pred 한 라벨
        return {"pred": pred, "label": items["labels"]}

    def test_epoch_end(self, outputs):
        pred_all = torch.concat([x["pred"] for x in outputs])
        label_all = torch.concat([x["label"] for x in outputs])
        test_f1 = metric.klue_re_micro_f1(pred_all.cpu().numpy(), label_all.cpu().numpy())
        self.log("test_micro_f1", test_f1)

    def predict_step(self, batch, batch_idx):
        items = batch
        logits = self(items)
        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lambda step: min(1.0, float(step + 1) / (self.warm_up + 1)))
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def freeze(self):
        for name, param in self.plm.named_parameters():
            freeze_list = []
            if name in freeze_list:
                param.requires_grad = False
