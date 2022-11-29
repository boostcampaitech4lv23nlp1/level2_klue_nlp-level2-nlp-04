import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import argparse


SEED = 123


def stratify(args):
    
    full_data = pd.read_csv(
        args.in_file,
        index_col=0
    )

    train, test = train_test_split(full_data, test_size=args.test_size, random_state=SEED, stratify=full_data['label'])

    train.to_csv(args.out_train, index=False)
    test.to_csv(args.out_test, index=False)

    print('num of train: \t', len(train))
    print('num of test: \t', len(test))


def check(args):

    train = pd.read_csv(args.out_train)
    test = pd.read_csv(args.out_test)

    train_cnt = Counter(train['label'])
    test_cnt = Counter(test['label'])

    train_ratio = {key: value/len(train) for key, value in train_cnt.items()}
    test_ratio = {key: value/len(test) for key, value in test_cnt.items()}

    print('label'+ " "*35+'|train_ratio|test_ratio|train_cnt|test_cnt|')
    for key in train_ratio:
        print(f"{key:40}|{train_ratio[key]:.4f}|{test_ratio[key]:.4f}|{train_cnt[key]:5}|{test_cnt[key]:5}|")



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Split train and test')

    parser.add_argument('--in-file', type=str, default='./train/train_drop_duplicates.csv',
                    help='Path of a Input File')
    parser.add_argument('--out-train', type=str, default='./train/train.csv',
                    help='Path of a Output File - Train Set')
    parser.add_argument('--out-test', type=str, default='./train/test.csv',
                    help='Path of a Output File - Test Set')
    parser.add_argument('--test-size', type=float, default=0.1,
                    help='Test Size (default: 0.1)')
    
    args = parser.parse_args()

    stratify(args)
    check(args)