import pandas as pd
from omegaconf import OmegaConf
import argparse


def reduce_no_relation(delete_num, train_df):
    print(f'no_relation 일부 삭제 전 train set 길이: {len(train_df)}')
    print(f'삭제할 no_relation 개수: {delete_num}')
    SEED=42
    delete_df = train_df[train_df['label'] == "no_relation"].sample(n=delete_num, random_state=SEED)
    delete_list = list(delete_df['id'])
    result = train_df[train_df['id'].apply(lambda x: x not in delete_list) == True].reset_index(drop=True)
    result.to_csv('train_reduce_no_relation.csv', index=False)

    print(f'삭제 후 train_set 길이: {len(result)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="base_config")
    parser.add_argument("--delete_num", "-n", type=int, default=4000)
    args, _ = parser.parse_known_args()
    conf = OmegaConf.load(f"../Config/{args.config}.yaml")
    train_df = pd.read_csv('../'+conf.path.train_path)
    # 삭제할 no_relation 개수 (EDA 결과 참고)
    delete_num = args.delete_num
    reduce_no_relation(delete_num, train_df)