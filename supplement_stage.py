#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Set, List
import networkx as nx
import requests


DEFAULT_BASE_URL = ""
DEFAULT_MODEL = ""


def _as_undirected_view(G: nx.Graph) -> nx.Graph:
    return G.to_undirected(as_view=True) if G.is_directed() else G


def _parse_comma_nodes(resp: str) -> Set[int]:
    """Parse comma-separated node ids.

    Defensive rules:
    - Empty/whitespace-only => empty set
    - Sentinel tokens like NONE/NULL/NIL => empty set
    - Ignores non-numeric stray tokens
    """
    if resp is None:
        return set()
    s = resp.strip()
    if not s:
        return set()
    up = s.upper()
    if up in {"NONE", "NULL", "NIL", "NO", "EMPTY"}:
        return set()

    out: Set[int] = set()
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        tok2 = tok.rstrip(". ")
        if tok2.lstrip("-").isdigit():
            out.add(int(tok2))
    return out


def _neighbors_1_2_hop_union(G: nx.Graph, C: Set[int], hop: int = 2) -> Set[int]:
    """返回 C 的 1-hop ∪ 2-hop 邻居（不含 C）"""
    Gv = _as_undirected_view(G)
    if not C:
        return set()

    n1: Set[int] = set()
    for u in C:
        if u in Gv:
            n1.update(Gv.neighbors(u))
    n1.difference_update(C)

    if hop <= 1:
        return n1

    n2: Set[int] = set()
    for x in n1:
        if x in Gv:
            n2.update(Gv.neighbors(x))
    n2.difference_update(C)

    return n1 | n2


def _compute_PN_by_deltaM(G_full: nx.Graph, C: Set[int], hop: int = 2) -> Set[int]:
    """
    ComGPT/LWP local modularity:
      M(C) = Min / Mout
    一次性扫描：
      PN = { v in N_{<=hop}(C) : M(C∪{v}) > M(C) }
    """
    if not C:
        return set()

    Gv = _as_undirected_view(G_full)

    # base Min/Mout
    Min = 0
    Mout = 0
    for u in C:
        if u not in Gv:
            continue
        for v in Gv.neighbors(u):
            if v in C:
                if u < v:
                    Min += 1
            else:
                Mout += 1

    M0 = float("inf") if Mout == 0 else (Min / Mout)

    cand = _neighbors_1_2_hop_union(Gv, C, hop=hop) - C
    PN: Set[int] = set()

    # 增量判断：
    # deg_in  = |N(v) ∩ C|
    # deg_out = deg(v) - deg_in
    # Min'  = Min + deg_in
    # Mout' = Mout - deg_in + deg_out
    for v in cand:
        if v not in Gv:
            continue

        deg_in = sum(1 for u in Gv.neighbors(v) if u in C)
        deg_v = Gv.degree(v)
        deg_out = deg_v - deg_in

        Min2 = Min + deg_in
        Mout2 = Mout - deg_in + deg_out
        M2 = float("inf") if Mout2 == 0 else (Min2 / Mout2)

        if M2 > M0:
            PN.add(v)

    return PN


def _build_comgpt_graph_text(G_full: nx.Graph, C: Set[int], PN: Set[int]) -> str:
    """
    ComGPT 风格 graph encoding（只用 C ∪ PN 的诱导子图结构）
    - Topology graph
    - Community information（C 与 PN 的连接摘要）
    """
    Gv = _as_undirected_view(G_full)
    V = set(C) | set(PN)
    nodes_sorted = sorted(V)

    lines: List[str] = []
    lines.append("Topology graph")
    lines.append(f"G describes a graph among nodes: {', '.join(map(str, nodes_sorted))}.")

    for u in nodes_sorted:
        nbrs = sorted([x for x in Gv.neighbors(u) if x in V]) if u in Gv else []
        lines.append(
            f"Node {u} is connected to nodes {', '.join(map(str, nbrs))}."
            if nbrs else
            f"Node {u} is connected to nodes null."
        )

    lines.append("")
    lines.append("Community information")
    lines.append(f"Nodes in the current community: [{', '.join(map(str, sorted(C)))}].")
    lines.append(f"The outside nodes contain: [{', '.join(map(str, sorted(PN)))}].")

    for v in sorted(PN):
        inC = sorted([x for x in Gv.neighbors(v) if x in C]) if v in Gv else []
        outPN = sorted([x for x in Gv.neighbors(v) if x in PN]) if v in Gv else []

        lines.append(
            f"Node {v} is connected to nodes within the community: [{', '.join(map(str, inC))}]."
            if inC else
            f"Node {v} is connected to nodes within the community: null."
        )
        lines.append(
            f"Node {v} is connected to nodes outside community: [{', '.join(map(str, outPN))}]."
            if outPN else
            f"Node {v} is connected to nodes outside community: null."
        )

    return "\n".join(lines)


def _build_prompt(G_full: nx.Graph, C: Set[int], PN: Set[int]) -> str:
    """Second-round supplement prompt.

    IMPORTANT: we only want the model to output **supplemental node ids** to add,
    instead of re-outputting/overwriting the whole community.

    Output format:
      - comma-separated ids, ONLY from PN
      - must NOT include nodes already in C
      - if nothing should be added, output: NONE
    """
    gtxt = _build_comgpt_graph_text(G_full, C, PN)

    return (
        "You are doing local community detection (supplement stage).\n"
        "You are given a current community C and a candidate outside set PN.\n"
        "PN is constructed from 1/2-hop neighbors of C and each node in PN increases local modularity M when added to C.\n\n"
        "Task:\n"
        "Select which nodes (if any) from PN should be ADDED to C based on the provided local subgraph structure.\n"
        "Please analyze whether these nodes should be added to the community C.\n"
        "The probability of not adding nodes is higher, but not always.\n"
        "If you think there is a suitable node, please output its node number.\n\n"
        "Graph data (topology + community information):\n"
        f"{gtxt}\n\n"
        "Output requirement (STRICT):\n"
        "- Output ONLY the node ids you want to ADD (supplemental ids), as a comma-separated list.\n"
        "- You MUST choose only from PN.\n"
        "- Do NOT output any node that is already in C.\n"
        "- If no node should be added, output exactly: NONE\n"
        "- No explanation, no extra words, no brackets.\n"
    )


def _load_api_key(api_key_path: str) -> str:
    with open(api_key_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _call_xiaoai_chat_completion(
    *,
    prompt: str,
    api_key_path: str = "apiKey.txt",
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1,
    timeout: int = 120,
) -> str:
    """
    按你 llmTest.py 的方式请求：
      POST {BASE_URL}/v1/chat/completions
    """
    api_key = _load_api_key(api_key_path)
    base_url = base_url.strip().rstrip("/")  # 你 llmTest.py 里 BASE_URL 有前导空格，这里做防呆
    url = f"{base_url}/v1/chat/completions"

    data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是一个图社区发现专家。\n"
                    "你必须直接给出最终答案，不得向用户提问，不得提供备选方案，不得输出解释或推理过程。\n"
                    "输出必须严格符合用户指定格式；除格式要求的内容外，不得输出任何额外文本。"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
    }

    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=data,
        timeout=timeout,
    )
    resp.raise_for_status()
    j = resp.json()
    return j["choices"][0]["message"]["content"]


# =========================
# Public API (ONE function)
# =========================
def supplement_once(
    *,
    G_full: nx.Graph,
    response1: str,
    hop: int = 2,
    api_key_path: str = "apiKey.txt",
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1,
    timeout: int = 120,
    fallback_to_first: bool = True,
) -> str:
    """
    - 第二轮 LLM 只输出补充节点 id（supplemental ids）
    - 函数内部做并集：C_final = C ∪ Add，确保不会覆盖第一轮
    """
    C = _parse_comma_nodes(response1)
    if not C:
        return response1

    PN = _compute_PN_by_deltaM(G_full, C, hop=hop)
    if not PN:
        return response1

    prompt2 = _build_prompt(G_full, C, PN)

    try:
        response2 = _call_xiaoai_chat_completion(
            prompt=prompt2,
            api_key_path=api_key_path,
            base_url=base_url,
            model=model,
            temperature=temperature,
            timeout=timeout,
        )

        add_raw = _parse_comma_nodes(response2)
        add = (add_raw & PN) - C   # ✅ 只允许补 PN，且不重复 C

        C_final = set(C) | set(add)
        return ",".join(map(str, sorted(C_final)))

    except Exception:
        return response1 if fallback_to_first else ""
