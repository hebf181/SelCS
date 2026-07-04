# SelCS
1.在apiKey.txt中写入api key
2.在llmTest.py中配置url,model等参数
3.运行命令如下:
  python pipeline.py \
  --embedding_path preTrain/pretrain_result/dolphins.npy \
  --edge_path comGPT_dataset/dolphins.gml \
  --query_nodes 42 \
  --mode local \
  --encoder incident \
  --pyg_pt_path preTrain/dataset/dolphins_pyg.pt \
  --id_map_path preTrain/dataset/dolphins_idmap.txt \
  --big_dataset \
  --candidate 1 \
  --prompt_version 13