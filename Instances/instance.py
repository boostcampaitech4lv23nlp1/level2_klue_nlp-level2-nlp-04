from Instances.Dataloaders.dataloader import Dataloader
from Instances.Dataloaders.k_fold_dataloader import KFoldDataloader
from Instances.Models.models import Model, BaseModel, ModelWithConcat, BinaryLoss, RBERT, RBERTWithLSTM, ModelWithLSTM, BaseModelWithPooling


def new_instance(conf):
    dataloader = Dataloader(conf)
    if conf.model.class_id == 0:
        model = Model(conf, dataloader.new_vocab_size())
    elif conf.model.class_id == 1:
        model = BaseModel(conf, dataloader.new_vocab_size())
        print(model)
    elif conf.model.class_id == 2:
        model = ModelWithConcat(conf, dataloader.new_vocab_size())
        print(model)
    elif conf.model.class_id == 3:
        if conf.data.entity_marker_type == "baseline":
            print("RBERT cannot be operated when 'baseline' is selected")
            exit(1)
        model = RBERT(conf, dataloader.new_vocab_size())
    elif conf.model.class_id == 4:
        if conf.data.entity_marker_type == "baseline":
            print("RBERT cannot be operated when 'baseline' is selected")
            exit(1)
        model = RBERTWithLSTM(conf, dataloader.new_vocab_size())
    elif conf.model.class_id == 5:
        model = ModelWithLSTM(conf, dataloader.new_vocab_size())
        print(model)
    elif conf.model.class_id == 6:
        model = BinaryLoss(conf, dataloader.new_vocab_size())
    elif conf.model.class_id == 7:
        model = BaseModelWithPooling(conf, dataloader.new_vocab_size())
    else:
        print("해당하는 모델이 없습니다")
        exit(1)
    return dataloader, model


def load_instance(args, conf):
    dataloader, model = new_instance(conf)
    save_path = "/".join(args.saved_model.split("/")[:-1])

    model_name = "/".join(args.saved_model.split("/")[1:-1]).split("_")[0]

    if args.saved_model.split(".")[-1] != "ckpt":
        exit("saved_model 파일 오류")

    model = model.load_from_checkpoint(args.saved_model)

    conf.path.save_path = save_path + "/"
    conf.model.model_name = "/".join(model_name.split("/")[1:])
    return dataloader, model, args, conf


def kfold_new_instance(conf, k):
    k_dataloader = KFoldDataloader(conf, k)
    if conf.model.class_id == 0:
        k_model = Model(conf, k_dataloader.new_vocab_size())
    elif conf.model.class_id == 1:
        k_model = BaseModel(conf, k_dataloader.new_vocab_size())
    elif conf.model.class_id == 2:
        k_model = ModelWithConcat(conf, k_dataloader.new_vocab_size())
    elif conf.model.class_id == 3:
        if conf.data.entity_marker_type == "baseline":
            print("RBERT cannot be operated when 'baseline' is selected")
            exit(1)
        k_model = RBERT(conf, k_dataloader.new_vocab_size())
    elif conf.model.class_id == 4:
        if conf.data.entity_marker_type == "baseline":
            print("RBERT cannot be operated when 'baseline' is selected")
            exit(1)
        k_model = RBERTWithLSTM(conf, k_dataloader.new_vocab_size())
    elif conf.model.class_id == 5:
        k_model = ModelWithLSTM(conf, k_dataloader.new_vocab_size())
    elif conf.model.class_id == 6:
        k_model = BinaryLoss(conf, k_dataloader.new_vocab_size())
    else:
        print("해당하는 모델이 없습니다")
        exit(1)

    return k_dataloader, k_model


def kfold_load_instance(args, conf, k):
    k_dataloader, k_model = kfold_new_instance(conf, k)

    model_name = "/".join(args.saved_model.split("/")[1:3])
    conf.model.model_name = model_name

    if args.saved_model.split(".")[-1] == "ckpt":
        print("saved_model 파일 오류, k_fold 설정 확인!")
        exit(1)
    k_model = k_model.load_from_checkpoint(args.saved_model + f"/{k+1}-Fold.ckpt")

    return k_dataloader, k_model
