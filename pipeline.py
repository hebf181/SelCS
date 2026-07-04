#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline.py  (放根目录, 最终极简版)

一条命令跑完：
1) custom_run: 分数 + 剪枝 + 搜索（仍可按你原习惯保存中间产物）
2) callLLM: 图转文本 + LLM 调用
3) 打印 + 保存 LLM 原始输出
4) 结束（不做任何指标/真值社区）

用法：
python pipeline.py \
  --embedding_path ./pretrain_result/cora.npy \
  --edge_path dataset/cora/cora.edges \
  --query_nodes 42,123 \
  --prune_mode conn_topk \
  --topk 2000 \
  --mode local \
  --encoder incident
"""

import argparse
import os
import networkx as nx
from networkx.algorithms.cuts import conductance
from networkx.algorithms.community.quality import modularity

import preTrain.custom_run as custom_run
from callLLM import llm_run
from eval_metric import evaluate_all_metrics

from bigDataset_llm import bigDataset_process
from supplement_stage import supplement_once

import os
import re
import torch

import os


def build_args():
    ap = argparse.ArgumentParser()

    # custom_run 参数（保持一致）
    ap.add_argument("--embedding_path", type=str, required=True)
    ap.add_argument("--edge_path", type=str, required=True)
    ap.add_argument("--query_nodes", type=str, required=True)
    ap.add_argument("--topk", type=int, default=2000)
    ap.add_argument("--prune_mode", type=str,
                    choices=["topk", "conn", "conn_topk"], default="topk")
    ap.add_argument("--mode", type=str,
                    choices=["global", "local"], default="global")
    ap.add_argument("--pyg_pt_path", type=str, default=None,
                help="Path to label .pt file that contains y (e.g., [A, X, y]). Used for truth community.")
    ap.add_argument("--id_map_path", type=str, default=None,
                help="two-column id map file; if provided, map edge_path's original ids to mapped 0..N-1 ids")
    ap.add_argument("--big_dataset", action="store_true",
                help="是否使用大数据集流程（加上该参数表示使用）")
    ap.add_argument("--candidate",type=int,default=1,help="是否使用候选集 1:使用 0:不使用")
    ap.add_argument("--prompt_version", type=int, default=1,
                help="选择使用哪一版prompt模板,默认为1")


    # LLM / 文本编码
    ap.add_argument("--encoder", type=str, default="incident",
                    help="TLAG 图文本编码器名，如 incident/adjacency/...")

    # 是否保存中间产物（默认 True 以不破坏你原习惯）
    ap.add_argument("--save_intermediate", action="store_true")
    ap.add_argument("--no_save_intermediate", action="store_true")

    return ap.parse_args()


def main():
    args = build_args()

    # 默认和你原来一样：保存中间产物
    save_intermediate = True
    if args.no_save_intermediate:
        save_intermediate = False
    elif args.save_intermediate:
        save_intermediate = True

    # Step1: 剪枝 + 分数 + 搜索
    (G_full, G_pruned, query_nodes, masked_scores,
     kept_nodes, base_comm, ranked) = custom_run.run_pruning_and_scoring(
        args, save_intermediate=save_intermediate, verbose=True
    )

    # scores txt 字符串（和原 txt 一致）
    score_str = custom_run.ranked_to_string(ranked)


    # Step2: 调 LLM（只要原始返回）
    if args.big_dataset == False:
        response = llm_run(
            G_pruned=G_pruned,
            score_str=score_str,
            query_nodes=query_nodes,
            encoder_name=args.encoder,
            version=args.prompt_version
        )

    # Step2.1: 大数据集走这条分支
    else:
        G_pruned =  bigDataset_process(args)
        ranked = custom_run.ranked_list_from_graph_embedding(G_pruned, args, args.query_nodes, args.embedding_path)
        # 节点相似度分数
        score_str = custom_run.ranked_to_string(ranked)
        response = llm_run(
            G_pruned=G_pruned,
            score_str=score_str,
            query_nodes=query_nodes,
            encoder_name=args.encoder,
            version=args.prompt_version
        )

    print("\n======= 第一轮社区结果 =======")
    print(response)
    print("================================\n")
    response = supplement_once(G_full=G_full, response1=response)
    print("\n======= 第二轮社区结果 =======")
    print(response)
    print("================================\n")

    # nodes = []
    nodes = [int(x.strip()) for x in response.split(",") if x.strip()]
    S=set(nodes)

    # 计算评估指标
    metrics = evaluate_all_metrics(
        G_full=G_full,
        pred_community=S,
        query_nodes=query_nodes,
        pyg_pt_path=args.pyg_pt_path,
        args = args
    )

    print("\n[Base community (search result) metrics on FULL graph]")
    print(f"  community_size        = {metrics['community_size']}")
    print(f"  conductance           = {metrics['conductance']}")
    print(f"  modularity            = {metrics['modularity']}")
    print(f"  internal_density      = {metrics['internal_density']}")
    print(f"  truth_community_size  = {metrics['truth_community_size']}")
    print(f"  TP                    = {metrics['TP']}")
    print(f"  FP                    = {metrics['FP']}")
    print(f"  FN                    = {metrics['FN']}")
    print(f"  precision             = {metrics['precision']}")
    print(f"  recall                = {metrics['recall']}")
    print(f"  f1                    = {metrics['f1']}")
    print("\nPipeline finished.")

if __name__ == "__main__":
    main()
