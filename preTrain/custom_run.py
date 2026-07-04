#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
custom_run.py
把“自定义查询 + 可调剪枝 (Top-K / 连通性)”接到原仓库的 GlobalSearch / LocalSearch 上。

在你的原始版本基础上：
- 保留 main() 以及脚本独立运行行为
- 新增 run_pruning_and_scoring(args, save_intermediate=True, verbose=True)
  供 pipeline.py 一次性调用。
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import os
import networkx as nx
import pickle


# -----------------------------------------------------------------------------
# ID 映射表缓存：支持把“原始节点 id”映射到“连续的 0..N-1 节点 id”
# 映射文件支持两列：
#   - new_id  old_id   （你截图这种：左边是 0..N-1，右边是原始 id）
#   - old_id  new_id   （反过来也支持，会自动识别）
# 空行/注释行(# 开头)会被忽略。
# -----------------------------------------------------------------------------
_ID_MAP_CACHE = {}

def _looks_like_0_to_n_minus_1(vals):
    if not vals:
        return False
    s = set(vals)
    if len(s) != len(vals):
        return False
    return (min(s) == 0) and (max(s) == len(s) - 1)

def load_id_map(id_map_path):
    """
    读取 id 对照表，返回 dict: old_id -> new_id。
    会做简单的列方向自动识别：若某一列形如 0..N-1，则认为它是 new_id 列。
    """
    if id_map_path is None:
        return None
    if id_map_path in _ID_MAP_CACHE:
        return _ID_MAP_CACHE[id_map_path]

    pairs = []
    with open(id_map_path, "r") as f:
        for line in f:
            line = line.strip()
            if (not line) or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            a, b = int(parts[0]), int(parts[1])
            pairs.append((a, b))

    if not pairs:
        raise ValueError(f"Empty id map file: {id_map_path}")

    col0 = [p[0] for p in pairs]
    col1 = [p[1] for p in pairs]

    col0_is_new = _looks_like_0_to_n_minus_1(col0)
    col1_is_new = _looks_like_0_to_n_minus_1(col1)

    # 默认按截图：col0=new, col1=old
    if col1_is_new and (not col0_is_new):
        # col1 是 new_id，则 col0 是 old_id
        old_to_new = {old: new for old, new in pairs}
    else:
        # col0 是 new_id，则 col1 是 old_id
        old_to_new = {old: new for new, old in pairs}

    _ID_MAP_CACHE[id_map_path] = old_to_new
    return old_to_new

# 剪枝后的 NetworkX 图,作为全局变量方便后续在本模块中使用
G = None


def parse_query_nodes(arg_str):
    """把形如 "42,123,7" 的字符串 -> [42,123,7]"""
    idxs = [s.strip() for s in arg_str.split(",") if s.strip() != ""]
    return [int(x) for x in idxs]


def load_embeddings(path):
    """从 .npy 里加载节点 embedding,返回 torch.Tensor [N, d]"""
    emb_np = np.load(path)
    emb = torch.from_numpy(emb_np).float()
    return emb


def calc_query_vec(embeddings, query_nodes):
    """
    embeddings: [N, d] tensor
    query_nodes: list[int]
    返回查询向量 q
    """
    q_emb = embeddings[query_nodes, :]           # [len(Q), d]
    q_vec = q_emb.mean(dim=0, keepdim=True)      # [1, d]
    return q_vec


def calc_scores(q_vec, embeddings):
    """
    q_vec: [1, d]
    embeddings: [N, d]
    计算每个节点和查询向量的余弦相似度 -> scores: [N]
    """
    q_rep = q_vec.expand_as(embeddings)          # [N, d]
    scores = F.cosine_similarity(q_rep, embeddings, dim=1)  # [N]
    return scores


def topk_mask(scores, topk):
    """
    scores: torch.Tensor [N]
    topk: int
    返回:
      masked_scores: torch.Tensor [N],除了 Top-K 保留原值,其它节点全设成 -1e9
      keep_idx: list[int],Top-K 节点索引(按分数高到低)
    """
    N = scores.shape[0]
    k = min(topk, N)

    top_vals, top_idx = torch.topk(scores, k=k, largest=True, sorted=True)
    keep_set = set(top_idx.tolist())

    masked_scores = scores.clone()
    neg_big = -1e9
    for nid in range(N):
        if nid not in keep_set:
            masked_scores[nid] = neg_big

    return masked_scores, top_idx.tolist()


def load_graph_from_edge_list(edge_path, num_nodes=None, node_id_base="auto", id_map_path=None):
    """
    从边列表文件加载无向图（NetworkX Graph）。

    支持：
      - .edge / .txt / .tsv 等：两列 u v（空格或 tab 分隔），支持以 # 开头的注释行
      - .gml：NetworkX 读取后转为 int 节点 id（常见 gml 节点是字符串形式的数字）

    重要：本脚本后续会用“节点 id 作为 embedding/scores 的下标”，因此图中的节点 id 必须落在 [0, num_nodes)。
    - 若提供 id_map_path：将 edge_path 中的“原始节点 id”映射为“连续的 0..N-1 节点 id”（old_id -> new_id），
      并以映射后的 id 建图。
      映射文件支持两列：new_id old_id（截图这种）或 old_id new_id（会自动识别）。
      在启用 id_map_path 时，会忽略 node_id_base 的自动 shift（shift 固定为 0），以避免把原始 id 平移后导致映射失败。
    - 若未提供 id_map_path：
        * node_id_base="auto" 且检测到 edge list 使用 1-based 编号 (最小为 1、最大为 num_nodes 且未出现 0)，则自动整体减 1。
        * 若仍有越界边（端点不在 [0, num_nodes)），将被跳过并记录。

    如果提供 num_nodes，则会确保 [0, num_nodes) 的节点都在图里（包括孤立点）。
    """
    ext = os.path.splitext(edge_path)[1].lower()

    # 0) 可选：加载 old_id -> new_id 映射
    old_to_new = load_id_map(id_map_path) if id_map_path else None
    use_id_map = old_to_new is not None

    # 1) 先扫描一遍 / 或读取图，做编号基准检测（仅在不使用 id_map 时生效）
    shift = 0
    min_id = None
    max_id = None
    has_zero = False

    if node_id_base not in ("auto", 0, 1):
        raise ValueError(f"node_id_base must be 'auto', 0 or 1, got: {node_id_base}")

    if use_id_map:
        # 有显式映射时，不做 shift（避免原始 id 被平移后映射不到）
        shift = 0
    else:
        if node_id_base == 1:
            shift = -1
        elif node_id_base == 0:
            shift = 0
        else:
            # auto: 仅在提供 num_nodes 时才做 1-based 自动识别
            if num_nodes is not None:
                if ext == ".gml":
                    G_tmp = nx.read_gml(edge_path, label="id")
                    try:
                        node_ids = [int(n) for n in G_tmp.nodes()]
                    except Exception as e:
                        raise ValueError(
                            "GML 节点 id 无法转换为 int（后续需要用作 embedding 下标）。"
                            f"请确认 gml 的 node label 是数字字符串。原始错误: {e}"
                        )
                    if node_ids:
                        min_id, max_id = min(node_ids), max(node_ids)
                        has_zero = (0 in node_ids)
                else:
                    with open(edge_path, "r") as f:
                        for line in f:
                            line = line.strip()
                            if not line or line.startswith("#"):
                                continue
                            parts = line.split()
                            if len(parts) < 2:
                                continue
                            u, v = int(parts[0]), int(parts[1])
                            if u == 0 or v == 0:
                                has_zero = True
                            if min_id is None:
                                min_id = min(u, v)
                                max_id = max(u, v)
                            else:
                                min_id = min(min_id, u, v)
                                max_id = max(max_id, u, v)

                # 典型 1-based：节点范围 [1, num_nodes]，且没出现 0
                if (min_id == 1) and (max_id == num_nodes) and (not has_zero):
                    shift = -1
                else:
                    shift = 0

    # 2) 真正建图，并跳过越界边
    G_full = nx.Graph()
    skipped_edges = 0
    skipped_unmapped = 0  # 使用 id_map 时：端点找不到映射的边

    if ext == ".gml":
        G_raw = nx.read_gml(edge_path, label="id")
        for u0, v0 in G_raw.edges():
            try:
                u_raw = int(u0)
                v_raw = int(v0)
            except Exception as e:
                raise ValueError(
                    "GML 边端点无法转换为 int（后续需要用作 embedding 下标）。"
                    f"请确认 gml 的 node label 是数字字符串。原始错误: {e}"
                )

            if use_id_map:
                if (u_raw not in old_to_new) or (v_raw not in old_to_new):
                    skipped_unmapped += 1
                    continue
                u, v = old_to_new[u_raw], old_to_new[v_raw]
            else:
                u, v = u_raw + shift, v_raw + shift

            if num_nodes is not None and not (0 <= u < num_nodes and 0 <= v < num_nodes):
                skipped_edges += 1
                continue
            G_full.add_edge(u, v)
    else:
        with open(edge_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue

                u_raw = int(parts[0])
                v_raw = int(parts[1])

                if use_id_map:
                    if (u_raw not in old_to_new) or (v_raw not in old_to_new):
                        skipped_unmapped += 1
                        continue
                    u, v = old_to_new[u_raw], old_to_new[v_raw]
                else:
                    u, v = u_raw + shift, v_raw + shift

                if num_nodes is not None:
                    if not (0 <= u < num_nodes and 0 <= v < num_nodes):
                        skipped_edges += 1
                        continue

                G_full.add_edge(u, v)

    if num_nodes is not None:
        for nid in range(num_nodes):
            if nid not in G_full:
                G_full.add_node(nid)

    # 记录下来，便于上层统一处理 query_nodes 的编号
    G_full.graph["id_shift"] = shift
    G_full.graph["skipped_edges"] = skipped_edges
    if use_id_map:
        G_full.graph["id_map_path"] = id_map_path
        G_full.graph["skipped_unmapped_edges"] = skipped_unmapped
    return G_full

def connectivity_mask(scores, G_full, query_nodes):
    """
    基于连通性剪枝：保留与任一查询点在同一连通分量里的所有节点。

    返回:
      masked_scores: torch.Tensor [N]
      kept_nodes: list[int]
    """
    N = scores.shape[0]
    keep_nodes = set()

    for q in query_nodes:
        if q in G_full:
            comp = nx.node_connected_component(G_full, q)
            keep_nodes.update(comp)

    if not keep_nodes:
        keep_nodes.update(query_nodes)

    masked_scores = scores.clone()
    neg_big = -1e9
    for nid in range(N):
        if nid not in keep_nodes:
            masked_scores[nid] = neg_big

    kept_nodes = sorted(keep_nodes)
    return masked_scores, kept_nodes


def build_pruned_graph(G_full, kept_nodes):
    """在原图 G_full 上,对 kept_nodes 做诱导子图,返回剪枝后的 NetworkX Graph。"""
    subG = G_full.subgraph(kept_nodes).copy()
    return subG


def ranked_list_from_community(community_nodes, masked_scores):
    """把社区节点按分数降序排列，返回 [(nid, score), ...]"""
    score_np = masked_scores.detach().cpu().numpy()
    N = score_np.shape[0]

    # 防御式处理：过滤越界节点，避免 IndexError
    valid_nodes = [nid for nid in community_nodes if 0 <= nid < N]
    if len(valid_nodes) != len(community_nodes):
        bad = [nid for nid in community_nodes if nid < 0 or nid >= N][:10]
        print(f"[WARN] Dropped {len(community_nodes) - len(valid_nodes)} node ids out of [0, {N-1}]. Examples: {bad}")

    ranked = sorted(
        [(nid, float(score_np[nid])) for nid in valid_nodes],
        key=lambda x: -x[1]
    )
    return ranked

# 新增函数：从图 + score获取排名列表
def ranked_list_from_graph_embedding(G, args, query, embedding_path):
    data = np.load(embedding_path, allow_pickle=False)
    emb = data[data.files[0]] if isinstance(data, np.lib.npyio.NpzFile) else data
    embeddings = torch.as_tensor(emb, dtype=torch.float32)  # [N, d]

    query = int(query)
    if getattr(args, "id_map_path", None):
        old_to_new = load_id_map(args.id_map_path)  # 参照你截图：old_id -> new_id

        if query not in old_to_new:
            raise ValueError(
                f"query_node(old_id)={query} 在映射表中找不到，请检查 id_map 文件是否包含该原始节点。"
            )
        query = old_to_new[query]  # new_id
    
    # query = int(query)
    q_vec = embeddings[query].unsqueeze(0)                  # [1, d]
    q_rep = q_vec.expand_as(embeddings)                     # [N, d]
    scores = F.cosine_similarity(q_rep, embeddings, dim=1)  # [N]

    score_np = scores.detach().cpu().numpy()
    ranked = sorted([(nid, float(score_np[int(nid)])) for nid in G.nodes()], key=lambda x: -x[1])
    return ranked

def ranked_to_string(ranked):
    """与原 txt 保存一致的字符串格式。"""
    return "".join([f"{nid}\t{sc:.6f}\n" for nid, sc in ranked])

def run_pruning_and_scoring(args, save_intermediate=True, verbose=True):

    # 1. 加载 embedding
    embeddings = load_embeddings(args.embedding_path)
    N = embeddings.shape[0]

    # 2. 加载原始图
    G_full = load_graph_from_edge_list(args.edge_path, num_nodes=N, id_map_path=getattr(args, 'id_map_path', None))

    # 3. 解析查询点
    query_nodes = parse_query_nodes(args.query_nodes)

    # ✅ 强制：传入的一定是“原始 old_id”，必须映射成“连续 new_id”
    if getattr(args, 'id_map_path', None):
        old_to_new = load_id_map(args.id_map_path)

        mapped_q = []
        for q in query_nodes:
            if q not in old_to_new:
                raise ValueError(f"query_node(old_id)={q} 在映射表中找不到，请检查 id_map 文件是否包含该原始节点。")
            mapped_q.append(old_to_new[q])

        query_nodes = mapped_q


    # 如果 edge list 被自动识别为 1-based 并做了 shift，这里同步修正 query_nodes
    id_shift = G_full.graph.get("id_shift", 0)
    if id_shift != 0:
        query_nodes = [q + id_shift for q in query_nodes]

    # 如果有越界边被跳过，提示一下（不影响运行，但说明 edge list/embedding 节点空间不一致）
    if verbose and G_full.graph.get("skipped_edges", 0) > 0:
        print(f"[WARN] Skipped {G_full.graph.get('skipped_edges')} edges with endpoints outside [0, {N-1}].")

    # 校验：query_nodes 必须在 [0, N) 内
    bad_q = [q for q in query_nodes if q < 0 or q >= N]
    if bad_q:
        raise ValueError(f"query_nodes out of range after id shift: {bad_q}. Expected each in [0, {N-1}].")


    # 4. 查询向量 + 分数
    q_vec = calc_query_vec(embeddings, query_nodes)
    scores = calc_scores(q_vec, embeddings)

    # 5. 剪枝掩码
    # prune_mode 可不传：默认 none => 不剪枝，保留全图
    if (getattr(args, "prune_mode", None) is None) or (args.prune_mode == "none"):
        masked_scores = scores
        kept_nodes = list(range(N))
    elif args.prune_mode == "topk":
        masked_scores, kept_nodes = topk_mask(scores, args.topk)
    elif args.prune_mode == "conn":
        masked_scores, kept_nodes = connectivity_mask(scores, G_full, query_nodes)
    else:
        conn_scores, conn_nodes = connectivity_mask(scores, G_full, query_nodes)
        masked_scores, kept_nodes = topk_mask(conn_scores, args.topk)
        # kept_nodes = kept_nodes ∩ conn_nodes（保持 kept_nodes 的 top 顺序）
        conn_set = set(conn_nodes)
        kept_nodes = [n for n in kept_nodes if n in conn_set]

        # 不在 kept_nodes 的全部设为 1e9
        keep_set = set(kept_nodes)
        big = -1e9
        for nid in range(N):
            if nid not in keep_set:
                masked_scores[nid] = big


    # 6. 构建剪枝子图
    global G
    G = build_pruned_graph(G_full, kept_nodes)
    G_pruned = G

    candidate_nodes = kept_nodes

    ranked = ranked_list_from_community(candidate_nodes, masked_scores)

    return G_full, G_pruned, query_nodes, masked_scores, kept_nodes, candidate_nodes, ranked


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--embedding_path", type=str, required=True,
                    help="预训练后的节点向量 .npy,例如 ./pretrain_result/cora.npy")
    ap.add_argument("--edge_path", type=str, required=True,
                    help="原始图的边列表文件 (.edge / 两列 u v 格式)。")
    ap.add_argument("--id_map_path", type=str, default=None,
                    help="可选：节点ID映射表路径(两列：new_id old_id 或 old_id new_id，脚本会自动识别列方向)。如果 edge_path 使用的是原始ID，而 embedding 使用的是映射后的 0..N-1 ID，请传这个参数。")
    ap.add_argument("--query_nodes", type=str, required=True,
                    help="查询节点ID列表,用逗号隔开,比如 '42,123' 或 '7'")
    ap.add_argument("--topk", type=int, default=2000,
                    help="Top-K剪枝时保留的节点数(prune_mode=topk 时生效)")
    ap.add_argument("--prune_mode", type=str,
                    choices=["none", "topk", "conn", "conn_topk"], default="none",
                    help="剪枝方式: none=不剪枝; topk=Top-K; conn=连通分量; conn_topk=先连通性再Top-K。")
    ap.add_argument("--mode", type=str, choices=["global", "local"], default="global",
                    help="用全局二分式扩团(global)还是贪心邻居扩团(local)。")
    ap.add_argument("--pyg_pt_path", type=str, default=None)
    ap.add_argument("big_dataset", type=int,default=0,
                help="是否使用大数据集流程,1表示是,0表示否")

    args = ap.parse_args()
    run_pruning_and_scoring(args, save_intermediate=True, verbose=True)


if __name__ == "__main__":
    main()
