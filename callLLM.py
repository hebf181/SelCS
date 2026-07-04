# callLLM.py  (极简版)
# 功能：把剪枝图 + 分数转文本后调用 LLM，并打印原始结果
# 不做社区解析/评估，你手动复制粘贴即可。

import os
import pickle
from pathlib import Path

from graphEncoder.talk_like_a_graph.graph_to_txt import graph_to_text_str
import llmTest


def load_pickle_graph(path):
    with open(path, "rb") as f:
        return pickle.load(f)
    
def llm_run(G_pruned, score_str, query_nodes, version, encoder_name="incident", llm_fn=None):
    """
    给 pipeline 用的极简函数：
    - 图转文本（incident by default）
    - 调 llmTest.query_neighbors
    - 返回原始 response（不解析）
    """
    if llm_fn is None:
        llm_fn = llmTest.query_neighbors

    # 图结构
    structured_txt = graph_to_text_str(G_pruned, encoder_name)

    # 调大模型
    response = llm_fn(version, structured_txt, score_str, query_nodes)
    return response

def main():
    import argparse
    ap = argparse.ArgumentParser()

    ap.add_argument("--graph_save_dir", type=str, default="preTrain/pruned_graphs")
    ap.add_argument("--graph_file", type=str, required=True)
    ap.add_argument("--score_path", type=str, required=True)
    ap.add_argument("--encoder", type=str, default="incident")
    ap.add_argument("--query_nodes", type=str, required=True)
    ap.add_argument("--prompt_version", type=int, default=13)
    

    args = ap.parse_args()

    graph_path = os.path.join(args.graph_save_dir, args.graph_file)
    G_pruned = load_pickle_graph(graph_path)

    with open(args.score_path, "r", encoding="utf-8") as f:
        score_str = f.read()

    query_nodes = [int(x) for x in args.query_nodes.split(",") if x.strip()]

    response = llm_run(
        G_pruned=G_pruned,
        score_str=score_str,
        query_nodes=query_nodes,
        version=args.prompt_version,
        encoder_name=args.encoder
    )

    print("\n======= LLM RAW RESPONSE =======")
    print(response)
    print("================================\n")


if __name__ == "__main__":
    main()
