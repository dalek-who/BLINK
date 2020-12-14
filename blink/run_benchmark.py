# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
DEBUG = True  # todo

import sys
sys.path.append(".")
sys.path.append("..")

import argparse
import prettytable

from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput
from pycallgraph import Config
from pycallgraph import GlobbingFilter


import blink.main_dense as main_dense
import blink.candidate_ranking.utils as utils

DATASETS = [

    {
        "name": "toy-10",
        "filename": "data/BLINK_benchmark/toy-10_AIDA-YAGO2_testa.jsonl",
    },
    # {
    #     "name": "AIDA-YAGO2 testa",
    #     "filename": "data/BLINK_benchmark/AIDA-YAGO2_testa.jsonl",
    # },
    # {
    #     "name": "AIDA-YAGO2 testb",
    #     "filename": "data/BLINK_benchmark/AIDA-YAGO2_testb.jsonl",
    # },
    # {"name": "ACE 2004", "filename": "data/BLINK_benchmark/ace2004_questions.jsonl"},
    # {"name": "aquaint", "filename": "data/BLINK_benchmark/aquaint_questions.jsonl"},
    # {
    #     "name": "clueweb - WNED-CWEB (CWEB)",
    #     "filename": "data/BLINK_benchmark/clueweb_questions.jsonl",
    # },
    # {"name": "msnbc", "filename": "data/BLINK_benchmark/msnbc_questions.jsonl"},
    # {
    #     "name": "wikipedia - WNED-WIKI (WIKI)",
    #     "filename": "data/BLINK_benchmark/wnedwiki_questions.jsonl",
    # },
]

PARAMETERS = {
    "faiss_index": None, # "flat",
    "index_path": None, # "models/faiss_flat_index.pkl",
    "test_entities": None,  # 默认用缓存好的wiki实体，如果用其他实体集（例如kbp或zeshel需要自行调整）
    "test_mentions": None,  # 这里设置为None，之后每次测试不同数据集时会赋值为相应数据集的地址
    "interactive": False,
    "biencoder_model": "models/biencoder_wiki_large.bin",
    "biencoder_config": "models/biencoder_wiki_large.json",
    "entity_catalogue": "models/entity.jsonl",
    "entity_encoding": "models/all_entities_large.t7",
    "crossencoder_model": "models/crossencoder_wiki_large.bin",
    "crossencoder_config": "models/crossencoder_wiki_large.json",
    "output_path": "output",
    "fast": False,
    "top_k": 100,
    # "lowercase": True,
    "debug": DEBUG,
    "preprocessed_kb_catalogue": "models/preprocessed_catalogue.json",  # None,
}
args = argparse.Namespace(**PARAMETERS)

logger = utils.get_logger(args.output_path)

models = main_dense.load_models(args, logger)
"""
    biencoder,
    biencoder_params,
    crossencoder,
    crossencoder_params,
    candidate_encoding,
    title2id,
    id2title,
    id2text,
    wikipedia_id2local_id,
    faiss_indexer,
"""

table = prettytable.PrettyTable(
    [
        "DATASET",
        "biencoder accuracy",
        "recall at 100",
        "crossencoder normalized accuracy",
        "overall unormalized accuracy",
        "support",
    ]
)

def main():
    for dataset in DATASETS:
        logger.info(dataset["name"])
        PARAMETERS["test_mentions"] = dataset["filename"]

        args = argparse.Namespace(**PARAMETERS)
        (
            biencoder_accuracy,
            recall_at,
            crossencoder_normalized_accuracy,
            overall_unormalized_accuracy,
            num_datapoints,
            predictions,
            scores,
        ) = main_dense.run(args, logger, *models)

        table.add_row(
            [
                dataset["name"],
                round(biencoder_accuracy, 4),
                round(recall_at, 4),
                round(crossencoder_normalized_accuracy, 4),
                round(overall_unormalized_accuracy, 4),
                num_datapoints,
            ]
        )

    logger.info("\n{}".format(table))


if __name__ == "__main__":
    main()
    # if utils.is_debug():
    #     main()
    # else:
    #     # 画函数调用图
    #     config = Config()
    #     config.trace_filter = GlobbingFilter(include=[
    #         '*'
    #     ])
    #     graphviz = GraphvizOutput()
    #     graph_dir = "/data/users/wangyuanzheng/projects/blink-home/function_call_graph/"
    #     graphviz.output_file = graph_dir + 'run_benchmark.png'
    #
    #     with PyCallGraph(output=graphviz, config=config):
    #         main()



