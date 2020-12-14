# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import torch
from tqdm import tqdm, trange
from sys import exit
from torch.utils.data import DataLoader, TensorDataset

# from pytorch_transformers.tokenization_bert import BertTokenizer

from blink.biencoder.zeshel_utils import world_to_id
from blink.common.params import ENT_START_TAG, ENT_END_TAG, ENT_TITLE_TAG


def select_field(data, key1, key2=None):
    if key2 is None:
        return [example[key1] for example in data]
    else:
        return [example[key1][key2] for example in data]


def get_context_representation(  # 这个比较复杂的tokenize过程，而不是直接用tokenizer.tokenize，是为了在需要truncate时，mention能放在比较中间的位置，左边、右边的context长度差不多
    sample,
    tokenizer,
    max_seq_length,
    # lowercase,
    mention_key="mention",
    context_key="context",
    ent_start_token=ENT_START_TAG,
    ent_end_token=ENT_END_TAG,
):
    mention = sample[mention_key]
    context_left = sample[context_key + "_left"]
    context_right = sample[context_key + "_right"]

    # if lowercase:
    #     mention = mention.lower()
    #     context_left = context_left.lower()
    #     context_right = context_right.lower()

    mention_tokens = []
    if sample[mention_key] and len(sample[mention_key]) > 0:
        mention_tokens = tokenizer.tokenize(mention)
        mention_tokens = [ent_start_token] + mention_tokens + [ent_end_token]

    context_left = tokenizer.tokenize(context_left)
    context_right = tokenizer.tokenize(context_right)

    left_quota = (max_seq_length - len(mention_tokens)) // 2 - 1  # mention左边的长度配额
    right_quota = max_seq_length - len(mention_tokens) - left_quota - 2  # mention右边长度配额。2是给cls和sep的
    left_add = len(context_left)
    right_add = len(context_right)
    if left_add <= left_quota:  # 依据左右实际长度，修正实际需要的配额
        if right_add > right_quota:
            right_quota += left_quota - left_add
    else:
        if right_add <= right_quota:
            left_quota += right_quota - right_add

    context_tokens = (
        context_left[-left_quota:] + mention_tokens + context_right[:right_quota]
    )

    context_tokens = ["[CLS]"] + context_tokens + ["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(context_tokens)
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    assert len(input_ids) == max_seq_length

    return {
        "tokens": context_tokens,
        "ids": input_ids,
    }


def get_candidate_representation(
    candidate_desc, 
    tokenizer, 
    max_seq_length,
    # lowercase,
    candidate_title=None,
    title_tag=ENT_TITLE_TAG,
):
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    # if lowercase:
    #     candidate_desc = candidate_desc.lower()
    cand_tokens = tokenizer.tokenize(candidate_desc)
    if candidate_title is not None:
        # if lowercase:
        #     candidate_title = candidate_title.lower()
        title_tokens = tokenizer.tokenize(candidate_title)
        cand_tokens = title_tokens + [title_tag] + cand_tokens

    cand_tokens = cand_tokens[: max_seq_length - 2]
    cand_tokens = [cls_token] + cand_tokens + [sep_token]

    input_ids = tokenizer.convert_tokens_to_ids(cand_tokens)
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    assert len(input_ids) == max_seq_length

    return {
        "tokens": cand_tokens,
        "ids": input_ids,
    }


def prepare_biencoder_data(
    samples,
    tokenizer,
    max_context_length,
    max_cand_length,
    silent,
    id2title,
    id2text,
    wikipedia_id2local_id: dict,
    # lowercase: bool,
    mention_key="mention",
    context_key="context",
    label_key="label_id",
    # title_key='Wikipedia_title',
    ent_start_token=ENT_START_TAG,
    ent_end_token=ENT_END_TAG,
    title_token=ENT_TITLE_TAG,
    debug=False,
    logger=None,
):
    processed_samples = []

    # if debug:
    #     samples = samples[:20]

    if silent:
        iter_ = samples
    else:
        iter_ = tqdm(samples)

    use_world = True

    for idx, sample in enumerate(iter_):
        context_tokens = get_context_representation(
            sample=sample,
            tokenizer=tokenizer,
            max_seq_length=max_context_length,
            mention_key=mention_key,
            context_key=context_key,
            ent_start_token=ent_start_token,
            ent_end_token=ent_end_token,
            # lowercase=lowercase
        )

        if "local_label_id" in sample:
            local_label_id = sample["local_label_id"]
        elif wikipedia_id2local_id and len(wikipedia_id2local_id) > 0:
            key_type = type(list(wikipedia_id2local_id.keys())[0])
            wikipedia_id = key_type(sample[label_key].strip())
            local_label_id = wikipedia_id2local_id[wikipedia_id]
            sample["local_label_id"] = local_label_id
        else:
            raise ValueError()
        label_tokens = get_candidate_representation(
                candidate_desc=id2text[local_label_id],
                tokenizer=tokenizer,
                max_seq_length=max_cand_length,
                candidate_title=id2title[local_label_id],
                # lowercase=lowercase,
            )

        record = {
            "context": context_tokens,
            "label": label_tokens,
            "label_idx": [local_label_id],
        }

        if "world" in sample:
            src = sample["world"]
            src = world_to_id[src]
            record["src"] = [src]
            use_world = True
        else:
            use_world = False

        processed_samples.append(record)

    if debug and logger:
        logger.info("====Processed samples: ====")
        for sample in processed_samples[:5]:
            logger.info("Context tokens : " + " ".join(sample["context"]["tokens"]))
            logger.info(
                "Context ids : " + " ".join([str(v) for v in sample["context"]["ids"]])
            )
            logger.info("Label tokens : " + " ".join(sample["label"]["tokens"]))
            logger.info(
                "Label ids : " + " ".join([str(v) for v in sample["label"]["ids"]])
            )
            logger.info("Src : %d" % sample["src"][0])
            logger.info("Label_id : %d" % sample["label_idx"][0])

    context_vecs = torch.tensor(
        select_field(processed_samples, "context", "ids"), dtype=torch.long,
    )
    cand_vecs = torch.tensor(
        select_field(processed_samples, "label", "ids"), dtype=torch.long,
    )
    if use_world:
        src_vecs = torch.tensor(
            select_field(processed_samples, "src"), dtype=torch.long,
        )
    label_idx = torch.tensor(
        select_field(processed_samples, "label_idx"), dtype=torch.long,
    )
    data = {
        "context_vecs": context_vecs,
        "cand_vecs": cand_vecs,
        "label_idx": label_idx,
    }

    if use_world:
        data["src"] = src_vecs
        tensor_data = TensorDataset(context_vecs, cand_vecs, src_vecs, label_idx)
    else:
        tensor_data = TensorDataset(context_vecs, cand_vecs, label_idx)
    return data, tensor_data
