import pandas as pd



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
    train_df = pd.read_csv('train/train.csv')
    # 삭제할 no_relation 개수 (EDA 결과 참고)
    delete_num = 4000
    reduce_no_relation(delete_num, train_df)