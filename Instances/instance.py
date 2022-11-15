from Instances.Dataloaders.dataloader import Dataloader
from Instances.Models.models import Model


def new_instance(conf):
    dataloader = Dataloader(conf)
    model = Model(conf, dataloader.new_vocab_size())
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
