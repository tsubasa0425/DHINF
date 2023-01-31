import torch
import dhg
import os
import pickle
from dataLoader import DataOption, get_cascades

def RelationGraph(data_name):
    data = DataOption(data_name)
    _u2idx = {}

    with open(data.u2idx_dict, 'rb') as f:
        _u2idx = pickle.load(f)

    if os.path.exists(data.net_data):
        with open(data.net_data, 'r') as f:
            edge_list = f.read().strip().split('\n')
            if data_name == 'douban' or data_name == 'twitter':
                edge_list = [edge.split(',') for edge in edge_list]
            else:
                edge_list = [edge.split(' ') for edge in edge_list]

            edge_list = [(_u2idx[edge[0]], _u2idx[edge[1]]) for edge in edge_list \
                         if edge[0] in _u2idx and edge[1] in _u2idx]
    else:
        return None

    user_size = len(_u2idx)
    relation_graph = dhg.Graph(user_size, edge_list, device=torch.device("cuda"))
    return relation_graph


def ConRelationHypergraph(data_name):

    # 建立关系图
    relation_graph = RelationGraph(data_name)
    # 根据关系图顶点的K阶邻居构建一个超图
    relation_hypergraph_0hop = dhg.Hypergraph.from_graph(relation_graph, device=torch.device("cuda"))
    relation_hypergraph_1hop = dhg.Hypergraph.from_graph_kHop(relation_graph, k=1, only_kHop=True, device=torch.device("cuda"))
    relation_hypergraph_2hop = dhg.Hypergraph.from_graph_kHop(relation_graph, k=2, only_kHop=True, device=torch.device("cuda"))
    return [relation_hypergraph_0hop, relation_hypergraph_1hop, relation_hypergraph_2hop]

def CascadeHypergraph(user_size, examples):
    examples = examples.tolist()
    edge_list = []
    for example in examples:
        example = set(example)
        if len(example) > 2:
            example.discard(0)
        edge_list.append(example)

    cascade_hypergraph = dhg.Hypergraph(user_size, edge_list, device=torch.device("cuda"))
    return cascade_hypergraph