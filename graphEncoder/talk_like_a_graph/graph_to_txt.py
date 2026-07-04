# import sys
# from pathlib import Path

# # 把 talk-like-a-graph 这一层加到 sys.path
# ROOT = Path(__file__).resolve().parents[1]  # talk-like-a-graph 目录
# if str(ROOT) not in sys.path:
#     sys.path.insert(0, str(ROOT))

# from talk_like_a_graph import graph_text_encoders
# import networkx as nx

import networkx as nx

from . import graph_text_encoders

def edge_to_graph(edge_file):
    """读取 .edge 文件并构建 NetworkX 图"""
    G = nx.Graph()
    with open(edge_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            u, v = parts[0], parts[1]
            try:
                u = int(u)
                v = int(v)
            except ValueError:
                pass
            G.add_edge(u, v)
    return G


def graph_to_text(graph, out_file: str, graph_encoder_name: str = "incident"):
    """
    调用 TLAG 提供的 encode_graph,将图转换为文本,并写入 out_file。
    """
    text = graph_text_encoders.encode_graph(
        graph,
        graph_encoder=graph_encoder_name
    )
    with open(out_file, 'w', encoding="utf-8") as f:
        f.write(text)

    print(f"编码完成！使用编码器 '{graph_encoder_name}'，已保存到 {out_file}")


# 新增：不落盘版，给 pipeline/callLLM 用
def graph_to_text_str(graph, graph_encoder_name: str = "incident"):
    """直接返回文本字符串，不写文件。"""
    text = graph_text_encoders.encode_graph(
        graph,
        graph_encoder=graph_encoder_name
    )
    return text


if __name__ == "__main__":
    edge_file = "preTrain/dataset/photo/photo.edges"
    out_file = "graphEncoder/txt_result/photo.txt"
    encoder_name = "incident"
    G = edge_to_graph(edge_file)
    graph_to_text(G, out_file, encoder_name)
