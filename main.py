import argparse
import random
import numpy as np
import torch
import pytorch_lightning as pl

from omegaconf import OmegaConf
from Step import train, inference, sweep, analysis


def init():  # args : 실행시 입력하는 인자, conf : yaml 파일에 저장된 하이퍼파라미터
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", "-m", required=True)
    parser.add_argument("--config", "-c", type=str, default="base_config")

    parser.add_argument(
        "--saved_model",
        "-s",
        default=None,
        help="저장된 모델의 파일 경로를 입력해주세요. 예시: SaveModels/klue/roberta-small_colorful-sweep-5/epoch=0-test_micro_f1=66.49635639313826.ckpt",
    )
    args, _ = parser.parse_known_args()
    conf = OmegaConf.load(f"./Config/{args.config}.yaml")

    # 재현성을 위한 시드 고정
    SEED = conf.utils.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    return args, conf


if __name__ == "__main__":

    (
        args,
        conf,
    ) = init()  # args는 초기에 시작하기 위한 인자를, conf는 경로 및 하이퍼파라미터를 작성한 yaml 파일을 사용합니다.

    if args.mode == "train" or args.mode == "t":
        if conf.k_fold.use_k_fold:  # num_folds 변수 확인, True라면 k폴드를 아니라면 일반 함수를 선택합니다
            train.k_fold_train(args, conf)
        else:
            train.train(args, conf)

    elif args.mode == "continue" or args.mode == "c":
        if args.saved_model is None:
            print("경로를 입력해주세요")
        elif conf.k_fold.use_k_fold:
            print("K-Fold 추가 학습 불가능!!")
        else:
            train.continue_train(args, conf)

    elif args.mode == "exp" or args.mode == "e":
        exp_count = int(input("실험할 횟수를 입력해주세요 "))
        sweep.sweep(args, conf, exp_count)

    elif args.mode == "inference" or args.mode == "i":
        if args.saved_model is None:
            print("경로를 입력해주세요")
        else:
            if conf.k_fold.use_k_fold:  # num_folds 변수 확인
                inference.k_fold_inference(args, conf)
            else:
                inference.inference(args, conf)

    elif args.mode == "analysis" or args.mode == "a":
        if args.saved_model is None:
            print("경로를 입력해주세요")
        else:
            analysis.cm_and_error(args, conf)

    else:
        print("모드를 다시 설정해주세요 ")
        print("train        : t,\ttrain")
        print("continue     : c,\tcontinue")
        print("sweep        : e,\texp")
        print("inference    : i,\tinference")
        print("analysis     : a,\tanalysis")
