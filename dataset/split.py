import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter


SEED = 123


def stratify(p=0.1):
    
    full_data = pd.read_csv(
        './train/train_drop_duplicates.csv',
        index_col=0
    )

    train, test = train_test_split(full_data, test_size=p, random_state=SEED, stratify=full_data['label'])

    train.to_csv('./train/train.csv', index=False)
    test.to_csv('./train/test.csv', index=False)

    print('train: \t', len(train))
    print('test: \t', len(test))


def check():
    train = pd.read_csv('./train/train.csv')
    test = pd.read_csv('./train/test.csv')

    train_cnt = Counter(train['label'])
    test_cnt = Counter(test['label'])

    train_ratio = {key: value/len(train) for key, value in train_cnt.items()}
    test_ratio = {key: value/len(test) for key, value in test_cnt.items()}


    for key in train_ratio:
        print(f"{key:40}|{train_ratio[key]:.4f}|{test_ratio[key]:.4f}|{train_cnt[key]:5}|{test_cnt[key]:5}|")



if __name__ == '__main__':
    stratify(0.1)
    check()