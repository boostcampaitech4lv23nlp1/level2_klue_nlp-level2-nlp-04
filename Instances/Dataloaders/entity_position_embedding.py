import torch


def get_entity_position_embedding(tokenizer, entity_marker_type, special_tokens, input_ids):
    token2id = {token: idx for token, idx in zip(tokenizer.all_special_tokens, tokenizer.all_special_ids)}
    subj_id, obj_id = [], []

    if entity_marker_type == "typed_entity_marker":
        for token in special_tokens:
            if token[1] == "S" or token[1:3] == "/S":
                subj_id.append(token2id[token])
            elif token[1] == "O" or token[1:3] == "/O":
                obj_id.append(token2id[token])

    elif entity_marker_type == "typed_entity_marker_punct":
        subj_id, obj_id = [token2id["@"]], [token2id["#"]]
        sub_marker_id = tokenizer.encode("+", add_special_tokens=False)[0]
        obj_marker_id = tokenizer.encode("^", add_special_tokens=False)[0]

    # subject/object special token(ex:'@', '#') 위치 정보를 담는 리스트 생성
    # e.g. subj_emb[0] = [13, 15], obj_emb[0] = [20, 22]
    subj_emb, obj_emb = [], []
    for ids in input_ids:
        subj_pos, obj_pos = [], []
        for i in range(len(ids)):
            if len(subj_pos) + len(obj_pos) == 4:
                break
            if ids[i] in subj_id:
                subj_pos.append(i)  # subject special token 위치 정보(시작, 끝)
            if ids[i] in obj_id:
                obj_pos.append(i)   # object special token 위치 정보(시작, 끝)
        subj_emb.append(subj_pos)
        obj_emb.append(obj_pos)

    """
    Entity Embeddings
    e.g. subj_embeddings[0] = [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
         obj_embeddings[0] = [0, 0, 0, 0, 0, 0, 1, 1, ... 0]
    """
    subj_embeddings, obj_embeddings = [], []
    if entity_marker_type == "typed_entity_marker": 
        for s_emb, o_emb in zip(subj_emb, obj_emb):
            temp_s_embeddings, temp_o_embeddings = [0] * len(input_ids[0]), [0] * len(input_ids[0])
            temp_s_embeddings[s_emb[0] + 1 : s_emb[1]] = [1] * (s_emb[1] - (s_emb[0] + 1)) # subject entity 위치는 1로 반환
            temp_o_embeddings[o_emb[0] + 1 : o_emb[1]] = [1] * (o_emb[1] - (o_emb[0] + 1)) # object entity 위치는 1로 반환
            subj_embeddings.append(temp_s_embeddings)
            obj_embeddings.append(temp_o_embeddings)

    elif entity_marker_type == "typed_entity_marker_punct":
        for s_emb, o_emb, ids in zip(subj_emb, obj_emb, input_ids):
            temp_s_embeddings, temp_o_embeddings = [0] * len(ids), [0] * len(ids)
            temp_smark_num = ids[s_emb[0] + 2 : s_emb[1]].tolist().index(sub_marker_id) + 1  # subject type marker "+" 위치 정보
            temp_omark_num = ids[o_emb[0] + 2 : o_emb[1]].tolist().index(obj_marker_id) + 1  # object type marker "^" 위치 정보
            temp_s_embeddings[s_emb[0] + 2 + temp_smark_num : s_emb[1]] = [1] * (s_emb[1] - (s_emb[0] + 2 + temp_smark_num)) # subject entity 위치는 1로 반환
            temp_o_embeddings[o_emb[0] + 2 + temp_omark_num : o_emb[1]] = [1] * (o_emb[1] - (o_emb[0] + 2 + temp_omark_num)) # object entity 위치는 1로 반환
            subj_embeddings.append(temp_s_embeddings)
            obj_embeddings.append(temp_o_embeddings)

    return torch.tensor(subj_embeddings), torch.tensor(obj_embeddings)
