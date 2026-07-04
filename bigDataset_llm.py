# 专门用来处理大数据集

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Hashable, List, Optional, Tuple

import networkx as nx
import argparse


from preTrain.custom_run import load_graph_from_edge_list
from preTrain.custom_run import load_embeddings
from preTrain.custom_run import load_id_map


def _approx_conductance_from_stats(cut: float, vol_c: float, total_vol: float) -> float:
    """phi(C) = cut(C, V\C) / min(vol(C), vol(V\C)). 用 total_vol=2|E| 计算 vol(V\C)."""
    denom = min(vol_c, max(total_vol - vol_c, 0.0))
    if denom <= 0:
        return 0.0
    return cut / denom


def _local_modularity_M(e_in: float, e_out: float) -> float:
    """ComGPT/Luo2008 的 local modularity M: M(C)=e_in(C)/e_out(C). e_out=external edges."""
    if e_out <= 0:
        return float("inf")
    return e_in / e_out

@dataclass
class PPRCandidateResult:
    G_cc: nx.Graph                 # 与 query 连通(或弱连通)的子图（复制后的）
    ppr: Dict[Hashable, float]     # 原始 PPR 分数
    penalized: Dict[Hashable, float]  # 度惩罚后的分数
    ranked_nodes: List[Hashable]   # 按 penalized 降序排序（包含 query）


def _get_cc_nodes_containing_query(G: nx.Graph, query: Hashable) -> set:
    """返回包含 query 的连通/弱连通分量节点集合。"""
    if query not in G:
        raise ValueError(f"query node {query!r} not in graph.")

    if G.is_directed():
        # 有向图：弱连通分量（忽略方向）
        for comp in nx.weakly_connected_components(G):
            if query in comp:
                return set(comp)
    else:
        # 无向图：连通分量
        for comp in nx.connected_components(G):
            if query in comp:
                return set(comp)

    # 理论上不会到这（query 必在某个分量里）
    return {query}


def ppr_rank_candidates(
    G: nx.Graph,
    query: Hashable,
    *,
    alpha: float = 0.85,
    beta: float = 1.0,
    weight: Optional[str] = "weight",
    max_iter: int = 100,
    tol: float = 1.0e-6,
) -> PPRCandidateResult:
    """
    基于 Personalized PageRank 生成候选排序（不截断）：
    - 只在与 query 连通(或弱连通)的子图上计算
    - 返回按 ppr/deg^beta 排序的节点列表（包含 query）
    """
    # 1) 仅保留与 query 连通的节点
    cc_nodes = _get_cc_nodes_containing_query(G, query)
    G_cc = G.subgraph(cc_nodes).copy()

    # 2) personalization：为保险起见给全量 dict（其它点 0，query 为 1）
    personalization = {n: 0.0 for n in G_cc.nodes()}
    personalization[query] = 1.0

    # 3) PPR（NetworkX pagerank）
    ppr = nx.pagerank(
        G_cc,
        alpha=alpha,
        personalization=personalization,
        weight=weight,
        max_iter=max_iter,
        tol=tol,
    )

    # 4) 超高 degree 惩罚：score = ppr / deg^beta
    #    这里 degree 用“无权度”（边条数）抑制 hub；你也可以改成 weighted degree
    penalized: Dict[Hashable, float] = {}
    for n in G_cc.nodes():
        deg = G_cc.degree(n)  # 无权度
        denom = (deg ** beta) if deg > 0 else 1.0
        penalized[n] = ppr.get(n, 0.0) / denom

    # 5) 排序（包含 query，不移除）
    ranked_nodes = sorted(G_cc.nodes(), key=lambda n: penalized.get(n, 0.0), reverse=True)

    return PPRCandidateResult(G_cc=G_cc, ppr=ppr, penalized=penalized, ranked_nodes=ranked_nodes)


def build_candidates_from_args(G: nx.Graph, args: Any) -> PPRCandidateResult:
    query = args.query_nodes
    if isinstance(query, str):
        query = int(query)

    # --- 按 id_map_path 把 query(old_id) 映射成 new_id ---
    id_map_path = getattr(args, "id_map_path", None)
    if id_map_path:
        old_to_new = load_id_map(id_map_path)  # 你 load_graph_from_edge_list 里用的同一个函数
        if old_to_new is not None:
            if query not in old_to_new:
                raise KeyError(f"query(old_id)={query} not found in id_map: {id_map_path}")
            query = old_to_new[query]  # 关键：把 query 转成图里的新 id

    return ppr_rank_candidates(
        G,
        query,
        alpha=0.85,
        beta=1.0,
        weight="weight",
    )

def getPPR(G_full, args):
    embeddings = load_embeddings(args.embedding_path)
    N = embeddings.shape[0]
    G_full = load_graph_from_edge_list(args.edge_path, num_nodes=N, id_map_path=getattr(args, 'id_map_path', None))
    result = build_candidates_from_args(G_full, args)
    return result.penalized


def bigDataset_process(args):
    if args is None:
        ap = argparse.ArgumentParser()
        ap.add_argument("--query_nodes", type=int, required=True)
        ap.add_argument("--edge_path", type=str, required=True)
        ap.add_argument("--embedding_path", type=str, required=True)
        ap.add_argument("--id_map_path", type=str, default=None)
        args = ap.parse_args()

    # 加载 embedding
    embeddings = load_embeddings(args.embedding_path)
    N = embeddings.shape[0]

    # 加载原始图
    G_full = load_graph_from_edge_list(args.edge_path, num_nodes=N, id_map_path=getattr(args, 'id_map_path', None))

    result = build_candidates_from_args(G_full, args)
    ranked = result.ranked_nodes        # 按惩罚后分数排序的节点list（包含query）
    G_cc = result.G_cc                  # 只保留与query连通(或弱连通)的子图
    ppr = result.ppr                    # 原始PPR分数 dict
    penalized = result.penalized        # 惩罚后分数 dict

    TOPK = 100   # 你原来就是 100
    HOPS = 1     # 这里改成 1/2/3 测试不同 hop
    # 取映射后的 query（和 build_candidates_from_args 里保持一致）
    q = args.query_nodes
    if isinstance(q, str):
        q = int(q)
    id_map_path = getattr(args, "id_map_path", None)
    if id_map_path:
        old_to_new = load_id_map(id_map_path)
        if old_to_new is not None:
            q = old_to_new[q]

    if args.candidate == 1:
        cand = ranked[:TOPK]  # 前100个候选节点
        G_pruned = result.G_cc.subgraph(cand).copy()
        return G_pruned
    elif args.candidate == 2:
        G_cc_view = G_cc.to_undirected(as_view=True) if G_cc.is_directed() else G_cc
        within = set(nx.single_source_shortest_path_length(G_cc_view, q, cutoff=HOPS).keys())

        # # 在 hop 邻域内按 ranked 取前 TOPK
        cand = [n for n in ranked if n in within][:TOPK]
        if q not in cand:
            cand.insert(0, q)

        G_pruned = G_cc.subgraph(cand).copy()

        # 只保留包含 query 的连通分量（去掉诱导后产生的“散点碎块”）
        G_pruned_view = G_pruned.to_undirected(as_view=True) if G_pruned.is_directed() else G_pruned
        if q in G_pruned_view:
            cc_nodes = set(nx.node_connected_component(G_pruned_view, q))
            G_pruned = G_pruned.subgraph(cc_nodes).copy()
        return G_pruned
    elif args.candidate == 3:
        G_view = G_full.to_undirected(as_view=True) if G_full.is_directed() else G_full
        nodes = set(nx.single_source_shortest_path_length(G_view, q, cutoff=HOPS).keys())
        G_pruned = G_full.subgraph(nodes).copy()
        return G_pruned
    elif args.candidate == 4:
        # 方法4：在 query 的 1-hop + 2-hop 邻域(HOP)内做贪心扩展
        # 目标：最小化“近似导度”(phi)；若 phi 相同，用 ComGPT 的 local modularity M 作为 tie-breaker
        # 说明：
        # - 仅在 G_full 上计算统计量（度数、cut、e_in）
        # - 扩展上限：TOPK（与方法1/2保持一致）
        G_view = G_full.to_undirected(as_view=True) if G_full.is_directed() else G_full

        # 1) HOP = query 的 2-hop 邻域（包含 query）
        hop_nodes = set(nx.single_source_shortest_path_length(G_view, q, cutoff=HOPS).keys())
        if q not in hop_nodes:
            hop_nodes.add(q)

        # 扩展上限：最多 TOPK 个点（包含 query）
        target_size = len(hop_nodes)
        if target_size <= 1:
            G_pruned = G_full.subgraph([q]).copy()
            return G_pruned

        # 2) 预计算度（只对 hop_nodes）
        deg = {n: G_view.degree(n) for n in hop_nodes}
        total_vol = float(2 * G_view.number_of_edges())  # undirected total volume = 2|E|

        # 3) 初始化 C={q} 的统计量
        C_set = {q}
        remaining = set(hop_nodes)
        remaining.discard(q)

        vol_c = float(deg.get(q, 0))
        cut = float(deg.get(q, 0))  # {q} 的外部边条数就是度
        e_in = 0.0

        # 维护“每个候选点 v 与当前 C 的连接数” count_in[v]
        count_in = {v: 0 for v in remaining}
        for nb in G_view.neighbors(q):
            if nb in remaining:
                count_in[nb] += 1

        # 记录 (C_nodes_list, phi, M)
        records: List[Tuple[List[int], float, float]] = []
        phi0 = _approx_conductance_from_stats(cut, vol_c, total_vol)
        M0 = _local_modularity_M(e_in, cut)
        records.append((sorted(C_set), phi0, M0))

        # 4) 逐步扩展：每次从 remaining 中选 1 个点加入
        #    选择规则：phi 最小；phi 相同则 M 最大；再相同选 id 最小（保证确定性）
        max_steps = target_size - 1
        for _step in range(max_steps):
            best_v = None
            best_phi = None
            best_M = None

            for v in remaining:
                in_edges = float(count_in.get(v, 0))
                dv = float(deg.get(v, 0))
                out_edges = dv - in_edges

                new_cut = cut + out_edges - in_edges
                new_vol_c = vol_c + dv
                new_phi = _approx_conductance_from_stats(new_cut, new_vol_c, total_vol)

                new_e_in = e_in + in_edges
                new_M = _local_modularity_M(new_e_in, new_cut)

                if best_v is None:
                    best_v, best_phi, best_M = v, new_phi, new_M
                else:
                    # 先比 phi（越小越好）
                    if new_phi < best_phi - 1e-12:
                        best_v, best_phi, best_M = v, new_phi, new_M
                    elif abs(new_phi - best_phi) <= 1e-12:
                        # phi 相同：比 M（越大越好）
                        if new_M > best_M + 1e-12:
                            best_v, best_phi, best_M = v, new_phi, new_M
                        elif abs(new_M - best_M) <= 1e-12 and v < best_v:
                            best_v, best_phi, best_M = v, new_phi, new_M

            if best_v is None:
                break

            # 5) 应用 best_v 的更新（增量更新 cut / vol_c / e_in / count_in）
            in_edges_best = float(count_in.get(best_v, 0))
            dv_best = float(deg.get(best_v, 0))
            out_edges_best = dv_best - in_edges_best

            # 更新统计量
            e_in += in_edges_best
            cut = cut + out_edges_best - in_edges_best
            vol_c += dv_best

            # 更新集合
            C_set.add(best_v)
            remaining.remove(best_v)
            count_in.pop(best_v, None)

            # 更新 count_in：best_v 的邻居与 C 的连接数 +1
            for nb in G_view.neighbors(best_v):
                if nb in remaining:
                    count_in[nb] = count_in.get(nb, 0) + 1

            # 记录当前状态
            cur_phi = _approx_conductance_from_stats(cut, vol_c, total_vol)
            cur_M = _local_modularity_M(e_in, cut)
            records.append((sorted(C_set), cur_phi, cur_M))

        # 6) 从 records 里选最优 C：phi 最小；phi 相同选 M 最大
        best_nodes, best_phi, best_M = records[0]
        for nodes, phi, Mv in records[1:]:
            if phi < best_phi - 1e-12:
                best_nodes, best_phi, best_M = nodes, phi, Mv
            elif abs(phi - best_phi) <= 1e-12 and Mv > best_M + 1e-12:
                best_nodes, best_phi, best_M = nodes, phi, Mv

        G_pruned = G_full.subgraph(best_nodes).copy()
        return G_pruned
    

        # 方法5:PPR-sweep（按 ranked 顺序做 sweep cut），取 conductance(phi) 最小；
    elif args.candidate == 5:
        G_view = G_cc.to_undirected(as_view=True) if G_cc.is_directed() else G_cc

        # 只在 q 的 HOPS-hop 邻域内做 sweep（按 ranked 顺序筛选）
        hop_nodes = set(nx.single_source_shortest_path_length(G_view, q, cutoff=HOPS).keys())
        hop_nodes.add(q)

        # 扩展顺序：q + ranked 中属于 HOPS-hop 邻域的点（去掉 q）
        order = [q] + [n for n in ranked if (n != q and n in hop_nodes)]

        if len(order) <= 1:
            G_pruned = G_full.subgraph([q]).copy()
            return G_pruned

        sweep_len = len(order)

        # 预计算度（在 sweep 的图上算）
        deg = {n: G_view.degree(n) for n in G_view.nodes()}
        total_vol = float(2 * G_view.number_of_edges())

        # 初始化 C={q}
        C_set = {q}
        vol_c = float(deg.get(q, 0))
        cut = float(deg.get(q, 0))
        e_in = 0.0

        def _count_in_edges_to_C(v: int) -> float:
            cnt = 0
            for nb in G_view.neighbors(v):
                if nb in C_set:
                    cnt += 1
            return float(cnt)

        # 记录所有 sweep 前缀集合：(nodes, phi, M)
        records: List[Tuple[List[int], float, float]] = []

        # 记录初始状态 C={q}
        phi0 = _approx_conductance_from_stats(cut, vol_c, total_vol)
        M0 = _local_modularity_M(e_in, cut)
        records.append((sorted(C_set), phi0, M0))

        # sweep：按 order 依次加入并记录每个前缀集合
        for i in range(1, sweep_len):
            v = order[i]
            if v in C_set:
                continue

            in_edges = _count_in_edges_to_C(v)
            dv = float(deg.get(v, 0))
            out_edges = dv - in_edges

            e_in += in_edges
            cut = cut + out_edges - in_edges
            vol_c += dv
            C_set.add(v)

            cur_phi = _approx_conductance_from_stats(cut, vol_c, total_vol)
            cur_M = _local_modularity_M(e_in, cut)
            records.append((sorted(C_set), cur_phi, cur_M))

        # 从所有 records 里选最优 C：phi 最小；phi 相同选 M 最大
        best_nodes, best_phi, best_M = records[0]
        for nodes, phi, Mv in records[1:]:
            if phi < best_phi - 1e-12:
                best_nodes, best_phi, best_M = nodes, phi, Mv
            elif abs(phi - best_phi) <= 1e-12 and Mv > best_M + 1e-12:
                best_nodes, best_phi, best_M = nodes, phi, Mv

        G_pruned = G_full.subgraph(best_nodes).copy()
        return G_pruned

if __name__ == "__main__":
    bigDataset_process()
