#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_metrics.py

根据你的定义：
- 真值社区 = 原图 G_full 中，查询节点所在连通分量的并集

本模块统一负责：
- 基于 G_full 和预测社区 pred_nodes 计算：
    - 导电率 conductance
    - 模块度 modularity
    - 内部密度 internal density
    - F1-score(pred_nodes vs truth community)
"""

from __future__ import annotations
from typing import Iterable, Dict, Any, Optional, Set

import networkx as nx
import os
import torch
from networkx.algorithms.cuts import conductance
from networkx.algorithms.community.quality import modularity
from bigDataset_llm import getPPR


# 相同label作为真值社区
# 如果查询节点集属于不同社区 那么分别取真值然后取交集
# def get_truth_community(
#     G_full: nx.Graph,
#     query_nodes: Iterable[int],
#     pyg_pt_path=None,
# ) -> Set[int]:
#     """
#     真值社区（label-based, 从 pyg .pt 读取 y）：
#     - 若能从 .pt 中读取到 y，则真值社区 = 与 query_nodes 同 label 的所有节点（多个 query 取并集）。
#     - 否则回退到旧逻辑：查询点所在连通分量并集。

#     .pt 路径获取优先级：
#     1) G_full.graph["pyg_pt_path"] / ["pt_path"] / ["label_pt_path"]
#     2) 环境变量 PYG_PT_PATH / LABEL_PT_PATH
#     """
#     y = None
#     pt_path = None

#     # G_full.graph["pyg_pt_path"] = "preTrain/dataset/cora_pyg.pt"
#     # G_full.graph["pyg_pt_path"] = "preTrain/dataset/photo_dgl.pt"
#     if pyg_pt_path:
#         G_full.graph["pyg_pt_path"] = pyg_pt_path


#     # 1) 从图属性拿 pt 路径
#     if hasattr(G_full, "graph"):
#         for key in ("pyg_pt_path", "pt_path", "label_pt_path"):
#             if key in G_full.graph:
#                 pt_path = G_full.graph[key]
#                 break

#     # 2) 从环境变量拿 pt 路径
#     if pt_path is None:
#         import os
#         pt_path = os.environ.get("PYG_PT_PATH") or os.environ.get("LABEL_PT_PATH")

#     # 3) 读 pt -> 取 y
#     if pt_path:
#         try:
#             import torch
#             obj = torch.load(pt_path, map_location="cpu")
#             # 你这个数据：obj 是 list=[A, X, y]
#             if isinstance(obj, (list, tuple)) and len(obj) >= 3:
#                 y = obj[2]
#             # 兼容：如果是 PyG Data / dict
#             elif hasattr(obj, "y"):
#                 y = obj.y
#             elif isinstance(obj, dict):
#                 for v in obj.values():
#                     if hasattr(v, "y"):
#                         y = v.y
#                         break
#         except Exception:
#             y = None

#     # 4) label-based 真值社区
#     if y is not None:
#         # torch.Tensor -> 可索引标量
#         if hasattr(y, "detach") and hasattr(y, "cpu"):
#             y = y.detach().cpu()

#         num_nodes = int(y.shape[0]) if hasattr(y, "shape") else len(y)
#         V = set(int(n) for n in G_full.nodes())

#         labels = set()
#         for q in query_nodes:
#             q = int(q)
#             if 0 <= q < num_nodes:
#                 lab = y[q]
#                 lab = int(lab.item()) if hasattr(lab, "item") else int(lab)
#                 labels.add(lab)

#         if not labels:
#             return set()

#         truth_nodes: Set[int] = set()
#         for i in range(num_nodes):
#             lab = y[i]
#             lab = int(lab.item()) if hasattr(lab, "item") else int(lab)
#             if lab in labels and i in V:
#                 truth_nodes.add(int(i))
#         return truth_nodes

#     # 5) 回退：连通分量并集（原逻辑不变）
#     truth_nodes_cc: Set[int] = set()
#     for q in query_nodes:
#         q_int = int(q)
#         if q_int in G_full:
#             comp = nx.node_connected_component(G_full, q_int)
#             truth_nodes_cc.update(comp)
#     return truth_nodes_cc

# 不从pt读取 一个节点可能存在两个不同社区
def get_truth_community(
    G_full: nx.Graph,
    query_nodes: Iterable[int],
    truth_cmty_path: str,
    idmap_path: str,
    *,
    verbose: bool = True,
) -> Set[int]:
    """
    真值社区（community-file-based, 支持重叠社区 + 去重重复行）：

    - 约束：query_nodes 只有 1 个查询点（new id 空间）
    - 不再从 pt 读取 y；不再使用连通分量回退
    - 输入的 query_nodes 和 G_full 都是“id 映射后的 new id 空间”
      因此需要：
        1) new_id -> old_id（用 idmap）
        2) 在 truth_cmty_path（old id 空间）里找所有包含 old_query 的社区行
           - 相同重复行只记一次（社区集合去重）
           - 若一个节点属于多个不同社区：这些社区都要纳入（最终返回取并集）
        3) 把最终社区成员 old_id -> new_id，再返回（保证与 G_full 的 id 空间一致）
    """

    # -----------------------------
    # 0) 解析 query_nodes（必须只有一个）
    # -----------------------------
    q_list = [int(x) for x in query_nodes]
    if len(q_list) != 1:
        raise ValueError(f"get_truth_community: query_nodes must contain exactly 1 node, got {len(q_list)}")
    q_new = q_list[0]

    # -----------------------------
    # 1) 读取 idmap：new_id -> old_id，并构建 old_id -> new_id
    #    约定：每行 "new_id<TAB>old_id" 或 "new_id old_id"
    # -----------------------------
    new2old = {}
    old2new = {}
    with open(idmap_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            n = int(parts[0])
            o = int(parts[1])
            new2old[n] = o
            # 若 old 出现多次（极少见），保留首次/最小 new（避免不确定性）
            if o not in old2new:
                old2new[o] = n

    if q_new not in new2old:
        raise KeyError(f"get_truth_community: q_new={q_new} not found in idmap: {idmap_path}")

    q_old = new2old[q_new]

    # -----------------------------
    # 2) 扫描真值社区文件（old id 空间），找所有包含 q_old 的社区
    #    - 去重：相同社区（同一组节点）只保留一次
    #    - 重叠：若 q_old 出现在多个不同社区，都保留
    # -----------------------------
    # 用 tuple(sorted(set(nodes))) 作为社区的 canonical key 来去重重复行
    matched_communities = set()  # set[tuple[int, ...]]

    with open(truth_cmty_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            # 该文件“每行一个社区”，默认整行都是节点 id
            try:
                nodes_old = [int(x) for x in parts]
            except Exception:
                # 某些行可能有异常字符，直接跳过
                continue

            if not nodes_old:
                continue

            # 注意：行内可能有重复节点，先 set
            node_set = set(nodes_old)
            if q_old not in node_set:
                continue

            key = tuple(sorted(node_set))
            matched_communities.add(key)

    if not matched_communities:
        if verbose:
            print(f"[get_truth_community] q_new={q_new} -> q_old={q_old}: no community matched in {truth_cmty_path}")
        return set()

    # -----------------------------
    # 3) 将匹配到的社区取并集，并映射回 new id 空间
    # -----------------------------
    truth_old_union: Set[int] = set()
    for comm in matched_communities:
        truth_old_union.update(comm)

    truth_new: Set[int] = set()
    missing_old = 0
    V_new = set(int(n) for n in G_full.nodes())  # 图本身的 new id 集合（再过滤一遍更安全）

    for o in truth_old_union:
        n = old2new.get(o)
        if n is None:
            missing_old += 1
            continue
        if n in V_new:
            truth_new.add(n)

    # -----------------------------
    # 4) 验证打印（建议你先开着，确认 q 的社区命中情况）
    # -----------------------------
    if verbose:
        sizes = sorted([len(comm) for comm in matched_communities])
        sample = sorted(list(truth_new))[:20]
        # print(
        #     f"[get_truth_community] q_new={q_new} -> q_old={q_old} | "
        #     f"matched_unique_comms={len(matched_communities)} comm_sizes={sizes[:10]}{'...' if len(sizes)>10 else ''} | "
        #     f"union_old={len(truth_old_union)} -> union_new_in_G={len(truth_new)} | "
        #     f"old_missing_in_idmap={missing_old} | sample_new={sample}"
        # )

    return truth_new

# 小数据集还是走这条
import os
from typing import Iterable, Optional, Set

import networkx as nx
import torch


def get_truth_community_from_pt(
    G_full: nx.Graph,
    query_nodes: Iterable[int],
    pyg_pt_path: Optional[str] = None,
    *,
    verbose: bool = True,
) -> Set[int]:
    """
    从 pt 里读出真值标签 y，并返回与 query_nodes 同标签的节点集合（并集）。

    兼容 pt 格式：
    1) torch.save([adj, x, y]) / (adj, x, y)   ✅ 你的 build_from_gml/_save_pyg_pt 常见
    2) torch.save({"y": y, ...})
    3) torch.save(Data(..., y=...)) 或 {"data": Data(...)}
    """
    # 1) pt 路径
    pt_path = (
        pyg_pt_path
        or G_full.graph.get("pyg_pt_path")
        or G_full.graph.get("pt_path")
        or G_full.graph.get("label_pt_path")
        or os.getenv("PYG_PT_PATH")
        or os.getenv("LABEL_PT_PATH")
    )
    if not pt_path:
        raise ValueError("get_truth_community_from_pt: pt_path is required (not found in args/graph/env).")

    # 2) load pt
    obj = torch.load(pt_path, map_location="cpu")

    # 3) 抽取 y（真值标签）
    y = None
    if isinstance(obj, (list, tuple)):
        if len(obj) >= 3:
            y = obj[2]
    elif isinstance(obj, dict):
        if "y" in obj:
            y = obj["y"]
        elif "data" in obj and hasattr(obj["data"], "y"):
            y = obj["data"].y
    else:
        if hasattr(obj, "y"):
            y = obj.y

    if y is None:
        raise KeyError(f"get_truth_community_from_pt: cannot find y in pt file: {pt_path}")

    if not isinstance(y, torch.Tensor):
        y = torch.as_tensor(y)
    y = y.view(-1).cpu()

    # 4) query nodes
    q_list = [int(x) for x in query_nodes]
    if not q_list:
        raise ValueError("get_truth_community_from_pt: query_nodes is empty.")

    V = set(int(n) for n in G_full.nodes())  # 安全过滤到图里存在的节点

    truth: Set[int] = set()
    for q in q_list:
        if q < 0 or q >= y.numel():
            raise IndexError(f"get_truth_community_from_pt: q={q} out of range [0, {y.numel()-1}]")

        q_label = int(y[q].item())
        if q_label < 0:
            raise ValueError(f"get_truth_community_from_pt: q={q} has unlabeled y[q]={q_label} | {pt_path}")

        idx = (y == q_label).nonzero(as_tuple=False).view(-1).tolist()
        truth.update(int(n) for n in idx if int(n) in V)

        if verbose:
            print(f"[get_truth_community_from_pt] q={q} label={q_label} -> matched_in_pt={len(idx)}", flush=True)

    if verbose:
        print(f"[get_truth_community_from_pt] truth_size_in_G={len(truth)} | pt={pt_path}", flush=True)

    return truth


def compute_structure_metrics(
    G_full: nx.Graph,
    community_nodes: Iterable[int],
) -> Dict[str, Optional[float]]:
    """
    计算结构性指标：
    - conductance
    - modularity（把 {社区, 其余节点} 当成两团）
    - internal_density
    """
    S = set(int(x) for x in community_nodes)
    if len(S) == 0:
        return {
            "conductance": None,
            "modularity": None,
            "internal_density": None,
        }

    V = set(G_full.nodes())
    # 导电率
    try:
        # 如果 S 等于全部节点，conductance 没意义，这里直接设为 None
        if S == V:
            phi = None
        else:
            phi = conductance(G_full, S)
    except Exception:
        phi = None

    # 模块度：{S, V\S}
    try:
        rest = V - S
        if len(rest) == 0:
            # 只有一个社区时，modularity 一般定义为 0，这里这么处理
            Q = 0.0
        else:
            communities = [S, rest]
            Q = modularity(G_full, communities)
    except Exception:
        Q = None

    # 内部密度：S 诱导子图的 density
    try:
        H = G_full.subgraph(S)
        d_internal = nx.density(H)
    except Exception:
        d_internal = None

    return {
        "conductance": phi,
        "modularity": Q,
        "internal_density": d_internal,
    }

def compute_tp_fp_fn(pred_nodes, truth_nodes):
    """
    计算 TP, FP, FN 并返回 Precision 和 Recall
    :param pred_nodes: 预测社区的节点集合
    :param truth_nodes: 真实社区的节点集合
    :return: TP, FP, FN, Precision, Recall
    """
    # 将节点集合转换为 set 类型，确保计算不重复
    P = set(pred_nodes)
    T = set(truth_nodes)

    # 计算交集（TP），预测中但不在真值中的节点（FP），真值中但不在预测中的节点（FN）
    TP = len(P & T)
    FP = len(P - T)
    FN = len(T - P)

    # 计算 Precision 和 Recall
    if TP + FP == 0:  # 避免除以0
        precision = 0
    else:
        precision = TP / (TP + FP)

    if TP + FN == 0:  # 避免除以0
        recall = 0
    else:
        recall = TP / (TP + FN)

    return TP, FP, FN, precision, recall



def compute_f1(
    pred_nodes: Iterable[int],
    truth_nodes: Iterable[int],
) -> Optional[float]:
    """
    基于节点集合计算 F1-score：
    precision = |P∩T| / |P|
    recall    = |P∩T| / |T|
    F1        = 2 * precision * recall / (precision + recall)
    """
    P = set(int(x) for x in pred_nodes)
    T = set(int(x) for x in truth_nodes)

    if len(P) == 0 or len(T) == 0:
        return None

    inter = P & T
    if len(P) == 0:
        return None
    precision = len(inter) / len(P)
    if len(T) == 0:
        return None
    recall = len(inter) / len(T)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def evaluate_all_metrics(
    G_full: nx.Graph,
    pred_community: Iterable[int],
    query_nodes: Iterable[int],
    args,
    pyg_pt_path=None,
) -> Dict[str, Any]:
    """
    pipeline 调用的统一入口。

    输入：
    - G_full         : 原图（NetworkX Graph）
    - pred_community : 预测社区节点集合（例如搜索算法得到的社区）
    - query_nodes    : 查询节点集合

    输出：
    - dict，包括：
        - community_size        : 预测社区大小
        - conductance
        - modularity
        - internal_density
        - truth_community_size  : 真值社区大小（连通分量并集）
        - f1                    : 预测社区 vs 真值社区 的 F1-score
    """
    pred_set = set(int(x) for x in pred_community)
    metrics: Dict[str, Any] = {
        "community_size": len(pred_set),
    }

    # 结构指标
    struct = compute_structure_metrics(G_full, pred_set)
    metrics.update(struct)

    # 真值社区（连通分量）
    if "amazon" in args.embedding_path:
        truth_cmty_path = "comGPT_dataset/com-amazon.top5000.cmty.txt"
        truth_set = get_truth_community(G_full, query_nodes, truth_cmty_path, args.id_map_path, verbose=False)
        metrics["truth_community_size"] = len(truth_set)
    elif "dblp" in args.embedding_path:
        truth_cmty_path = "comGPT_dataset/com-dblp.top5000.cmty.txt"
        truth_set = get_truth_community(G_full, query_nodes, truth_cmty_path, args.id_map_path, verbose=False)
        metrics["truth_community_size"] = len(truth_set)
    else:
        truth_set = get_truth_community_from_pt(G_full, query_nodes, pyg_pt_path=pyg_pt_path, verbose=False)
        metrics["truth_community_size"] = len(truth_set)
    

    if len(truth_set) == 0:
        metrics["f1"] = None
        return metrics

    # F1-score
    f1 = compute_f1(pred_set, truth_set)
    metrics["f1"] = f1

    # Precision & Recall
    TP, FP, FN, precision, recall = compute_tp_fp_fn(pred_set, truth_set)
    metrics["precision"] = precision
    metrics["recall"] = recall
    metrics["TP"] = TP
    metrics["FP"] = FP
    metrics["FN"] = FN

    metrics["PPR"] = getPPR(G_full, args)
    return metrics
