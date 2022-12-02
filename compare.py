import pandas as pd
import re
from ast import literal_eval

df = pd.DataFrame()


file_name = "submission_clear-pond-47_epoch=3"

predict_data = pd.read_csv("./dataset/test/test_data.csv")
origin_data = pd.read_csv("./output.csv")
new_data = pd.read_csv(f"./prediction/{file_name}.csv")

subj_df = predict_data["subject_entity"].apply(lambda x: pd.Series(literal_eval(x))).add_prefix("subj_")
obj_df = predict_data["object_entity"].apply(lambda x: pd.Series(literal_eval(x))).add_prefix("obj_")

subj_df.reset_index(drop=True, inplace=True)
obj_df.reset_index(drop=True, inplace=True)
all_dataset = predict_data[["id", "sentence", "label"]]
all_dataset.reset_index(drop=True, inplace=True)
preprocessing_dataframe = pd.concat([all_dataset, subj_df, obj_df], axis=1)

sents = []
for sent, subj, subj_start, subj_end, subj_type, obj, obj_start, obj_end, obj_type in zip(
    preprocessing_dataframe["sentence"],
    preprocessing_dataframe["subj_word"],
    preprocessing_dataframe["subj_start_idx"],
    preprocessing_dataframe["subj_end_idx"],
    preprocessing_dataframe["subj_type"],
    preprocessing_dataframe["obj_word"],
    preprocessing_dataframe["obj_start_idx"],
    preprocessing_dataframe["obj_end_idx"],
    preprocessing_dataframe["obj_type"],
):
    temp_subj = f"@{str(subj)}@"
    temp_obj = f"#{str(obj)}#"

    if subj_start < obj_start:
        sent = sent[:subj_start] + temp_subj + sent[subj_end + 1 : obj_start] + temp_obj + sent[obj_end + 1 :]
    else:
        sent = sent[:obj_start] + temp_obj + sent[obj_end + 1 : subj_start] + temp_subj + sent[subj_end + 1 :]
    sents.append(sent)
df["id"] = predict_data["id"]
df["sentence"] = sents
df["subj"] = preprocessing_dataframe["subj_word"]
df["obj"] = preprocessing_dataframe["obj_word"]
df["origin_data"] = origin_data["pred_label"]
df["new_data"] = new_data["pred_label"]
index = df["origin_data"] != df["new_data"]

df.loc[index].to_csv(f"./compare.csv", index=False)
