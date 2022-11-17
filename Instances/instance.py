from Instances.Dataloaders.dataloader import Dataloader
from Instances.Models.models import Model, ExampleModel1, ExampleModel2


def new_instance(conf):
    dataloader = Dataloader(conf)
    if conf.model.class_id == 0:
        model = Model(conf, dataloader.new_vocab_size())
    elif conf.model.class_id == 1:
        model = ExampleModel1(conf, dataloader.new_vocab_size())
        print(model)
    elif conf.model.class_id == 2:
        model = ExampleModel2(conf, dataloader.new_vocab_size())
        print(model)
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
