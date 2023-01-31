import operator

from dataLoader import *
from util import *
from hypergraphConstruct import *
from Module import *
from torch.utils.data import DataLoader
import argparse
import torch.optim as optim
import torch
from time import time
from Metrics import *

parser = argparse.ArgumentParser()
parser.add_argument('-dataset_name', default='christianity')
parser.add_argument('-epoch', default=50)
parser.add_argument('-batch_size', default=64)
parser.add_argument('-emb_dim', default=128)
parser.add_argument('-train_rate', default=0.8)
parser.add_argument('-valid_rate', default=0.1)
parser.add_argument('-max_seq_length', default=200)
parser.add_argument('-lr', default=0.001)
parser.add_argument('-lambda_a', default=0.1)
parser.add_argument('-lambda_u', default=0.04)
parser.add_argument('-early_stop_step', default=10)

opt = parser.parse_args()

def train_epoch(opt, examples, masks, targets, model, relation_hypergraph, user_size, device, cascade_hypergraph):
    examples = examples.to(device)
    masks = masks.to(device)
    pred, co_attn_wts, user_loss = model(relation_hypergraph, cascade_hypergraph, examples, masks, opt.lambda_u)
    labels_k_hot = one_hot(targets, user_size).to(device)

    loss = torch.mean(labels_k_hot * (-torch.log(
        torch.maximum(torch.sigmoid(pred), torch.ones((pred.shape[0], pred.shape[1])).to(device) * (
            1e-10)))) * user_size / opt.max_seq_length + (1 - labels_k_hot) * (-torch.log(
        torch.maximum(1 - torch.sigmoid(pred),
                      torch.ones((pred.shape[0], pred.shape[1])).to(device) * (1e-10))))) \
           + opt.lambda_a * torch.sum(co_attn_wts ** 2)

    loss += user_loss
    return loss

def train(relation_hypergraph):
    dataset = opt.dataset_name
    max_seq_length = opt.max_seq_length
    batch_size = opt.batch_size
    emb_dim = opt.emb_dim
    lambda_a = opt.lambda_a
    lambda_u = opt.lambda_u
    epoch = opt.epoch
    lr = opt.lr
    early_stop_step = opt.early_stop_step
    patience = early_stop_step
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = Cascades(dataset, max_seq_length, mode='train')
    valid_data = Cascades(dataset, max_seq_length, mode='valid')
    test_data = Cascades(dataset, max_seq_length, mode='test')
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    user_size = len(relation_hypergraph[0].v)

    model = DHINF(user_size, emb_dim).cuda()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    scores_metrics = None
    score = float('-inf')
    best_epoch = 0
    total_time = 0
    for epoch in range(epoch):

        model.train()

        losses = []
        epoch_time = 0
        for i, (examples, targets, masks) in enumerate(train_loader, 0):

            cascade_hypergraph = CascadeHypergraph(user_size, examples)
            batch_start = time()
            loss = train_epoch(opt, examples, masks, targets, model, relation_hypergraph, user_size, device, cascade_hypergraph=cascade_hypergraph)
            batch_end = time()
            epoch_time += batch_end-batch_start

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        epoch_loss = np.mean(losses)
        print(f'Mean Prediction loss at epoch{epoch+1}: {epoch_loss}')
        print(f'Train time at epoch{epoch+1}: {epoch_time} second')


        # 开始验证
        model.eval()
        losses = []
        for i, (examples, targets, masks) in enumerate(valid_loader, 0):
            cascade_hypergraph = CascadeHypergraph(user_size, examples)
            loss = train_epoch(opt, examples, masks, targets, model, relation_hypergraph, user_size, device, cascade_hypergraph=cascade_hypergraph)
            losses.append(loss.item())

        epoch_loss = np.mean(losses)
        print(f'Mean Validation Prediction loss at epoch{epoch+1}: {epoch_loss}')



        # 开始测试
        k_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

        model.eval()

        total_samples = 0
        num_eval_k = len(k_list)
        avg_map_scores, avg_recall_scores = [0.]*num_eval_k, [0.]*num_eval_k
        for i, (examples, targets, masks) in enumerate(test_loader, 0):

            examples = examples.to(device)
            masks = masks.to(device)
            cascade_hypergraph = CascadeHypergraph(user_size, examples)
            pred, co_attn_wts, user_loss = model(relation_hypergraph, cascade_hypergraph, examples, masks, lambda_u)
            top_k = torch.topk(pred, 200, dim=1).indices
            top_k_filter = remove_seeds(top_k.detach().cpu().numpy(), examples.cpu().numpy())
            masks = get_masks(top_k_filter, examples.cpu())
            relevance_scores_all = get_relevance_scores(top_k_filter, targets.cpu().numpy())
            m = torch.sum(one_hot(targets.cpu(), user_size), dim=-1).numpy()
            relevance_scores = masked_select(relevance_scores_all, masks)

            recall_scores = [mean_recall_at_k(relevance_scores, k, m) for k in k_list]
            map_scores = [MAP(relevance_scores, k, m) for k in k_list]
            num_samples = relevance_scores.shape[0]
            avg_map_scores = list(
                map(operator.add, map(operator.mul, map_scores, [num_samples]*num_eval_k), avg_map_scores))
            avg_recall_scores = list(map(operator.add, map(operator.mul, recall_scores, [num_samples]*num_eval_k), avg_recall_scores))

            total_samples += num_samples

        avg_map_scores = list(map(operator.truediv, avg_map_scores, [total_samples] * num_eval_k))
        avg_recall_scores = list(map(operator.truediv, avg_recall_scores, [total_samples] * num_eval_k))
        metrics = dict()
        for k in range(0, num_eval_k):
            K = k_list[k]
            metrics[f"MAP@{K}"] = avg_map_scores[k]
            metrics[f"Recall@{K}"] = avg_recall_scores[k]
        # 打印本轮测试结果
        print(metrics)
        print()
        total_time += epoch_time
        if avg_map_scores[-1] > score:
            score = avg_map_scores[-1]
            scores_metrics = metrics
            best_epoch = epoch+1
            patience = early_stop_step
        else:
            patience -= 1

        if patience == 0:
            print('early stop!')
            break

    print(f'=============== best_result ===============')
    print(f'epoch: {best_epoch}')
    print(f'result:\n{scores_metrics}')
    print(f'Total time: {total_time}')



if __name__ == '__main__':
    dataset = opt.dataset_name
    relation_hypergraph_list = ConRelationHypergraph(dataset)
    train(relation_hypergraph_list)


