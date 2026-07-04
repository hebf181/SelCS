import requests
import json
from typing import List, Union
from pathlib import Path

BASE_URL = ""
MODEL = ""



with open("apiKey.txt", "r", encoding="utf-8") as f:
    API_KEY = f.read().strip()


def query_neighbors(version, structured_text: str, similarity_text: str, query_nodes: List[Union[str, int]]) -> str: 
    # 把所有查询点打印成 "1016, 2020, 3050" 这种形式 
    query_nodes_str = ", ".join(str(n) for n in query_nodes) 
    # 图结构+相似度
    if version == 13:
        content = ( 
            f"查询节点的编号为：[{query_nodes_str}]。\n" 
            "请你根据图结构和相似度信息,推断哪些顶点应该与查询节点集合属于同一社区,最后列出的所有的社区节点集合,以逗号分隔不要加其他任何字符" \
            "社区是一个在图中彼此连接较为紧密、与图中其他节点连接相对较少的节点集合。你可以把它理解成社交网络中的“小圈子”：圈子内部成员之间有较多连接关系,对外部成员的连接相对较少。" 
            "【节点与查询节点集合的相似度分数】\n" f"{similarity_text}\n"
            "【图的文本编码】\n" f"{structured_text}\n"
            )
        data = {
            "model": MODEL, 
            "messages": [
                {
                    "role": "system", 
                    "content": "你是一个图社区发现专家,你需要根据图的结构和相似度信息找出与查询节点集合属于同一社区的所有节点.\n"
                    "你必须直接给出最终答案，不得向用户提问，不得提供备选方案，不得输出解释或推理过程。\n"
                    "输出必须严格符合用户指定格式；除格式要求的内容外，不得输出任何额外文本。"
                },
                { 
                    "role": "user", 
                    "content": content, 
                }
            ],
            "temperature":0.1
        }
    # 图结构
    elif version == 15:
        content = ( 
            f"查询节点的编号为：[{query_nodes_str}]。\n" 
            "请你根据图结构信息,推断哪些顶点应该与查询节点集合属于同一社区,最后列出的所有的社区节点集合,以逗号分隔不要加其他任何字符" \
            "社区是一个在图中彼此连接较为紧密、与图中其他节点连接相对较少的节点集合。你可以把它理解成社交网络中的“小圈子”：圈子内部成员之间有较多连接关系,对外部成员的连接相对较少。" \
            "【图的文本编码】\n" f"{structured_text}\n"
            )
        data = {
            "model": MODEL, 
            "messages": [
                {
                    "role": "system", 
                    "content": "你是一个图社区发现专家,你需要根据图结构信息找出与查询节点集合属于同一社区的所有节点.\n"
                    "你必须直接给出最终答案，不得向用户提问，不得提供备选方案，不得输出解释或推理过程。\n"
                    "输出必须严格符合用户指定格式；除格式要求的内容外，不得输出任何额外文本。"
                },
                { 
                    "role": "user", 
                    "content": content, 
                }
            ],
            "temperature":0.1
        }
    # 相似度
    elif version == 16:
        content = ( 
            f"查询节点的编号为：[{query_nodes_str}]。\n" 
            "请你根据每个节点与查询节点的相似度信息,推断哪些顶点应该与查询节点集合属于同一社区,最后列出的所有的社区节点集合,以逗号分隔不要加其他任何字符" \
            "社区是一个在图中彼此连接较为紧密、与图中其他节点连接相对较少的节点集合。你可以把它理解成社交网络中的“小圈子”：圈子内部成员之间有较多连接关系,对外部成员的连接相对较少。" \
            "与查询节点相似度越大的节点,更可能与查询节点在同一个社区\n"
            "【节点与查询节点集合的相似度分数】\n" f"{similarity_text}\n"
            )
        data = {
            "model": MODEL, 
            "messages": [
                {
                    "role": "system", 
                    "content": "你是一个图社区发现专家,你需要根据相似度信息找出与查询节点集合属于同一社区的所有节点.\n"
                    "你必须直接给出最终答案，不得向用户提问，不得提供备选方案，不得输出解释或推理过程。\n"
                    "输出必须严格符合用户指定格式；除格式要求的内容外，不得输出任何额外文本。"
                },
                { 
                    "role": "user", 
                    "content": content, 
                }
            ],
            "temperature":0.1
        }

    resp = requests.post( f"{BASE_URL}/v1/chat/completions", headers={ "Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json", }, json=data, ) 
    resp.raise_for_status()
    resp_dict = resp.json()
    return resp_dict["choices"][0]["message"]["content"]