from scipy.stats import t
from sklearn.linear_model import LogisticRegression
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn import metrics
import os
import glob
from utils import getprotoconfi,getknnconfi,protoPred,knn_st,statistic,statistic2,addImages,addImages3,addImages2,class_balance,calculate_accuracy,isThan16,stastic_balance,statistic_balance
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.weight_norm import WeightNorm
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import scipy as sp
# import faiss
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear( indim, outdim, bias = False)
        self.class_wise_learnable_norm = True  #See the issue#4&8 in the github
        if self.class_wise_learnable_norm:
            WeightNorm.apply(self.L, 'weight', dim=0) #split the weight update component to direction and norm

        self.scale_factor = 10 #in omniglot, a larger scale factor is required to handle >1000 output classes.

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm+ 0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized) #matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm, see the issue#4&8 in the github
        scores = self.scale_factor* (cos_dist)

        return scores


def update_plabels(support, support_ys, query,query_ys,pnknn_scores):
    if isinstance(query_ys, torch.Tensor):
        query_ys = query_ys.cpu()  # 将 tensor 复制到 CPU
        query_ys = query_ys.numpy()  # 将 tensor 转换为 NumPy 数组
    max_iter = 20
    # 确定类别的数量
    no_classes = support_ys.max() + 1
    # 将支持集和查询集的特征合并，并确保数组是C连续的，以避免后续操作中可能出现的错误。
    X = np.concatenate((support, query), axis=0).copy(order='C')# to bypass the error of array not C-contiguous
    # X = F.normalize(X, p=2, dim=1, eps=1e-12)

    # 计算每个向量的 L2 范数
    norms = np.linalg.norm(X, axis=1, keepdims=True)

    # 归一化每个向量
    X = X / norms
    # 从配置选项中获取近邻的数量K。
    k = 5
    # if opt.model == 'resnet12':
    #     k = X.shape[0]-1
    alpha = 0.8
    # 初始化一个全零的标签数组labels，然后将支持集的标签复制到这个数组的前部分。
    labels = np.zeros(X.shape[0])
    labels[:support_ys.shape[0]]= support_ys
    labels[support_ys.shape[0]:] = query_ys
    # 创建两个索引数组，分别对应标记和未标记样本的索引。
    labeled_idx = np.arange(support.shape[0])
    unlabeled_idx = np.arange(query.shape[0]) + support.shape[0]

    all_label_idx =np.concatenate((labeled_idx, unlabeled_idx))

    nearest_neighbors = NearestNeighbors(n_neighbors=k + 2, metric='cosine').fit(X)
    distances, indices = nearest_neighbors.kneighbors(X)

    rows, cols = distances.shape

    D1 = np.zeros((rows, cols - 1))
    I1 = np.zeros((rows, cols - 1))

    for i in range(distances.shape[0]):
        if round(distances[i, 0], k + 2) == 0:
            D1[i, :] = distances[i, 1:k + 2]
            I1[i, :] = indices[i, 1:k + 2]
        else:
            D1[i, :] = distances[i, 0:k+1]
            I1[i, :] = indices[i, 0:k+1]



    D = np.zeros((rows, cols - 2))
    I = np.zeros((rows, cols - 2))

    for i in range(D1.shape[0]):
        if round(D1[i, 0], k + 1) == 0:
            D[i, :] = D1[i, 1:k + 1]
            I[i, :] = I1[i, 1:k + 1]
        else:
            D[i, :] = D1[i, 0:k]
            I[i, :] = I1[i, 0:k]



# ------------------------------
    # # 搜索每个样本的k+1个最近邻（包括自身），D是距离，I是索引。
    N = X.shape[0]


    # 构建一个图，其中节点之间的权重是基于最近邻的距离。
    # 这行代码创建了一个从0到N-1的整数数组row_idx，其中N是样本的数量。这个数组将用于构建图的行索引。
    row_idx = np.arange(N)
    # 结果是一个k行N列的二维数组，其中每一行都是row_idx的副本。
    row_idx_rep = np.tile(row_idx, (k, 1)).T
    # 这行代码创建了一个大小为N x N的稀疏矩阵W，其中W[i, j]表示样本i和样本j之间的权重。这个矩阵是对称的，因为每个样本与其最近邻之间的权重是相互的。
    # sp.sparse.csr_matrixs的三个参数分别是非零元素的值，行索引和列索引
    W = sp.sparse.csr_matrix((D.flatten('F'), (row_idx_rep.flatten('F'), I.flatten('F'))), shape=(N, N))
    # 这行代码通过将矩阵W与其转置W.T相加来创建一个对称矩阵。这是因为在无向图中，如果节点i与节点j相连，则节点j也与节点i相连。对称矩阵确保了这种相互连接。
    W = W + W.T

    # 对图进行归一化处理。
    # Normalize the graph
    # 这行代码从W中移除对角线元素，即每个节点到自身的权重（自环）。sp.sparse.diags函数根据给定的对角线元素创建一个对角矩阵，然后从W中减去这个矩阵
    W = W - sp.sparse.diags(W.diagonal())
    # 这行代码计算W中每一行的和，即每个节点的度（与该节点相连的边的权重之和）。
    S = W.sum(axis=1)
    # 这行代码将度为零的节点的度设置为1。这是必要的，因为后续的归一化步骤涉及到除以度，而除以零是未定义的。
    S[S == 0] = 1
    # 这行代码计算每个节点的归一化因子，即每个节点度的平方根的倒数。
    D = np.array(1. / np.sqrt(S))
    # 这行代码将归一化因子数组D转换为对角矩阵，其中对角线上的元素是归一化因子。
    D = sp.sparse.diags(D.reshape(-1))
    Wn = D * W * D

    # 初始化一个用于存储每个样本属于每个类别的概率的矩阵Z，并构建拉普拉斯矩阵A。
    # Initiliaze the y vector for each class (eq 5 from the paper, normalized with the class size) and apply label propagation
    Z = np.zeros((N, int(no_classes.item())))
    # 这行代码构建图的拉普拉斯矩阵A。拉普拉斯矩阵定义为单位矩阵I减去归一化权重矩阵Wn乘以一个参数alpha。alpha是一个超参数，用于控制图的平滑度
    # 拉普拉斯矩阵通常定义为度矩阵 D 减去权重矩阵 W
    A = sp.sparse.eye(Wn.shape[0]) - alpha * Wn
    # # 对于每个类别，使用共轭梯度法（CG）来解线性系统，更新Z矩阵。
    for i in range(int(no_classes.item())):
        cur_idx = all_label_idx[np.where(labels[all_label_idx] == i)]
        y = np.zeros((N,))
        y[cur_idx] = 1.0

        for abc in range(query_ys.shape[0]):
            y[len(support)+abc] = pnknn_scores[abc,i]

        f, _ = sp.sparse.linalg.cg(A, y, tol=1e-6, maxiter=max_iter)
        Z[:, i] = f



    # 将Z矩阵中的负值设置为0。
    # Handle numberical errors
    Z[Z < 0] = 0

    # --------try to filter-----------
    # 计算未标记样本的Z矩阵的最大值，并取负号
    # z_amax =-1*np.amax(Z, 1)[support_ys.shape[0]:]

    z_amax = 0

    #-----------trying filtering--------
    # Compute the weight for each instance based on the entropy (eq 11 from the paper)
    # 计算每个样本属于每个类别的概率，并将负概率设置为0。然后找到概率最大的类别作为预测标签，并计算最大概率值。
    probs_l1 = F.normalize(torch.tensor(Z), 1).numpy()
    # probs_l1 = F.normalize(Z.clone().detach(), 1).numpy()
    probs_l1[probs_l1 < 0] = 0
    p_labels = np.argmax(probs_l1, 1)
    p_probs = np.amax(probs_l1,1)

    # 将标记样本的预测标签设置为实际标签。
    p_labels[labeled_idx] = labels[labeled_idx]

    # 返回未标记样本的预测标签、概率以及z_amax值。
    return p_labels[support.shape[0]:], probs_l1, z_amax #p_probs #weights[support.shape[0]:]

def weight_imprinting(X, Y, model):
    no_classes = Y.max()+1
    imprinted = torch.zeros(no_classes, X.shape[1])
    for i in range(no_classes):
        idx = np.where(Y == i)
        tmp = torch.mean(X[idx], dim=0)
        tmp = tmp/tmp.norm(p=2)
        imprinted[i, :] = tmp
    model.weight.data = imprinted
    return model



def compute_optimal_transport(no_samples, M, epsilon=1e-6):

    unbalanced = False
    sinkhorn_iter = 1
    T = 3
    #r is the P we discussed in paper r.shape = n_runs x total_queries, all entries = 1
    r = torch.ones(1, M.shape[0])
    #r = r * weights
    #c = torch.ones(1, M.shape[1]) * int(M.shape[0]/M.shape[1])
    # c是目标概率分布
    c = torch.FloatTensor(no_samples)
    idx = np.where(c.detach().cpu().numpy() <= 0)
    if unbalanced == True:
        c = torch.FloatTensor(no_samples)
        idx = np.where(c.detach().cpu().numpy() <=0)
        if len(idx[0])>0:
            M[:, idx[0]] = torch.zeros(M.shape[0], 1)

    M = M.cuda()
    r = r.cuda()
    c = c.cuda()
    M = torch.unsqueeze(M, dim=0)
    n_runs, n, m = M.shape
    P = M

    u = torch.zeros(n_runs, n).cuda()
    maxiters = 1000
    iters = 1
    for i in range(sinkhorn_iter):
        P = torch.pow(P, T)
        while torch.max(torch.abs(u - P.sum(2))) > epsilon:
            u = P.sum(2)
            P *= (r / u).view((n_runs, -1, 1))
            P *= (c / P.sum(1)).view((n_runs, 1, -1))
            if len(idx[0]) > 0:
                P[P != P] = 0
            if iters == maxiters:
                break
            iters = iters + 1
    P = torch.squeeze(P).detach().cpu().numpy()
    best_per_class = np.argmax(P, 0)
    if M.shape[1]==1:
        P = np.expand_dims(P, axis=0)
    labels = np.argmax(P, 1)
    return P, labels, best_per_class

def rank_per_class(no_cls, rank, ys_pred, no_keep):
    list_indices = []
    list_ys = []
    for i in range(no_cls):
        cur_idx = np.where(ys_pred == i)
        y = np.ones((no_cls,))*i
        class_rank = rank[cur_idx]
        class_rank_sorted = sp.stats.rankdata(class_rank, method='ordinal')
        class_rank_sorted[class_rank_sorted > no_keep] = 0
        indices = np.nonzero(class_rank_sorted)
        list_indices.append(cur_idx[0][indices[0]])
        list_ys.append(y)
    idxs = np.concatenate(list_indices, axis=0)
    ys = np.concatenate(list_ys, axis = 0)
    return idxs, ys


def produce_labal(proto_pred,knn_pred,indices,support_ys,n_queries,n_lsamples,support_features,query_features):
    addImage, addLabel = addImages(proto_pred, knn_pred, indices, support_ys.cuda())
    labelnum = n_queries
    temp = n_lsamples
    addLabel2 = addImages2(support_features, support_ys.cuda(), query_features,
                           labelnum, temp)

    addLabel2Index = []
    for addLabel2Index1, addLabel2Index2 in enumerate(addLabel2):
        if (addLabel2Index2 > -1):
            addLabel2Index.append(addLabel2Index1)
    add_label = [-1 for _ in range(labelnum)]

    for i5 in range(len(addImage)):
        add_label[addImage[i5]] = addLabel[i5].item()

    add_label3 = add_label.copy()
    for q, w in enumerate(add_label):

        if (w == -1):
            if (addLabel2[q] <= -1):
                continue
            else:
                add_label3[q] = addLabel2[q]
        else:
            if (addLabel2[q] <= -1):
                continue
            else:
                if (addLabel2[q] == w):
                    continue
                else:
                    add_label3[q] = -2

    addImage_pro = []
    addLabel_pro = []

    for x1 in range(len(add_label3)):
        if (add_label3[x1] > -1):
            addImage_pro.append(x1)
            addLabel_pro.append(add_label3[x1])

    if len(addImage_pro) != 0:

        proto_support_images = torch.cat((support_features, query_features[addImage_pro]), dim=0)

        addLabel_tensor = torch.tensor(addLabel_pro)
        addLabel_tensor2 = torch.tensor(addLabel_pro)
        proto_support_labels = torch.cat((support_ys, addLabel_tensor.cuda()), dim=0).cuda()

    else:
        proto_support_images = support_features
        proto_support_labels = support_ys.cuda()
    return query_features[addImage_pro] , addLabel_tensor

def iter_balanced_trans(support_features, support_ys, query_features, classifier,real_label):

    n_lsamples = support_ys.shape[0]
    n_usamples = query_features.shape[0]
    n_shot = support_ys.shape[0]
    n_queries = query_features.shape[0]
    labels = torch.arange(5).repeat(int((n_lsamples/5)))

    # no_samples代表预测标签均匀分布，每个类有75/5个，一共五个类
    no_samples = np.array(np.repeat(float(query_features.shape[0] / 5), 5))

    # 查询集样本数作为迭代次数

    new_pred3 = torch.full((n_queries,), -1)
    new_pred4 = torch.full((n_queries,), -1)

    # 计算伪标签---------------------------------------------
    score_pre, indices_pre, knn_distances_pre, knn_pred_pre, scores_pre = classifier(support_features, support_ys, query_features,support_features, support_ys, 5)
    proto_pred_pre = protoPred(score_pre, support_ys)

    support_features_pre = torch.cat((support_features, query_features), dim=0)
    support_labal_pre = torch.cat((support_ys, proto_pred_pre[0]), dim=0)

    score, indices, knn_distances, knn_pred, scores = classifier(support_features_pre, support_labal_pre, query_features,support_features, support_ys, 5)
    proto_pred = protoPred(score, support_ys)
    new_pred2 = proto_pred[0]

    # -----------------------------------------------------------------------------------------------
    query_features_pred = torch.tensor([])
    query_label_pred = torch.tensor([])

    support_features2 = torch.empty(0)
    support_ys2 = torch.empty(0)
    proto_pred_updata = []
    knn_pred_updata = []
    indices_updata = []
    iter_acc = []
    fintune_label = proto_pred[0]
    lp_pred = proto_pred[0]
    for i in range(5):



        # if(support_ys2.size(0)<=support_ys.size(0)):
        #     persudo_image_idx, persudo_labal = addImages(proto_pred, knn_pred, indices, support_labal_pre.cuda())
        # else:
        #     persudo_image_idx, persudo_labal = addImages(proto_pred_updata, knn_pred_updata, indices_updata, support_ys2.cuda())


        if (support_ys2.size(0) <= support_ys.size(0)):
            persudo_image_idx, persudo_labal = addImages3(proto_pred, knn_pred, indices, support_labal_pre.cuda(),lp_pred)
        else:
            persudo_image_idx, persudo_labal = addImages3(proto_pred_updata, knn_pred_updata, indices_updata,
                                                         support_ys2.cuda(),lp_pred)

        persudo_labal = torch.tensor(persudo_labal)
        persudo_image = query_features[persudo_image_idx]
        a = 1.0
        pnknn_scores = scores.clone()
        pnknn_scores *= a
        for pnknn_scores_line in range(pnknn_scores.shape[0]):
            for knn_distances_row in range(knn_distances.shape[1]):
                temp = indices[pnknn_scores_line, knn_distances_row] % pnknn_scores.shape[1]
                pnknn_scores[pnknn_scores_line, temp] += (0) * (knn_distances[pnknn_scores_line][knn_distances_row])

        # 计算每一行的总和
        row_sums = pnknn_scores.sum(dim=1, keepdim=True)
        # 进行行归一化
        pnknn_scores = pnknn_scores / row_sums

        support_features2 = torch.cat((support_features, persudo_image), dim=0)
        support_features_numpy = support_features2.detach().cpu().numpy()
        query_features_numpy = query_features.detach().cpu().numpy()
        support_ys2 = torch.cat((support_ys, persudo_labal.cuda()), dim=0).long()
        support_ys_copy = support_ys2.clone().cpu()

        # 调用update_plabels函数来更新查询集的预测标签，返回预测标签、概率和权重。
        query_ys_pred, probs, weights = update_plabels(support_features_numpy, support_ys_copy, query_features_numpy,fintune_label,pnknn_scores)

        new_pred2 = query_ys_pred

        # 使用compute_optimal_transport函数计算最优传输矩阵，并更新预测标签和索引。
        P, new_pred3, indices_cot = compute_optimal_transport(no_samples, torch.Tensor(probs[n_lsamples+persudo_image.size(0):]))

        # _, n, m = P.shape
        n, m = probs.shape
        r = torch.ones(n_lsamples + n_usamples, device='cuda')
        c = torch.ones(5, device='cuda') * (n_shot + n_queries)
        u = torch.zeros(n, device='cuda')
        maxiters = 1000
        iters = 1
        # normalize this matrix
        probs[n_lsamples+persudo_image.size(0):] = P
        P_tensor = torch.from_numpy(probs).cuda()

        while torch.max(torch.abs(u - P_tensor.sum(dim=1))) > 0.01:
            u = P_tensor.sum(dim=1)
            # P_tensor *= (r / u).view((-1, 1))
            # P_tensor *= (c / P_tensor.sum(dim=0)).view((1, -1))

            # 行归一化：使每一行的和为1
            P_tensor = P_tensor / P_tensor.sum(dim=1, keepdim=True)
            # 列归一化：使每一列的和为1
            P_tensor = (P_tensor / P_tensor.sum(dim=0, keepdim=True))*(n_shot + n_queries)

            P_tensor[:n_lsamples].fill_(0)
            P_tensor[:n_lsamples].scatter_(1, labels[:n_lsamples].unsqueeze(1).cuda(), 1)
            if iters == maxiters:
                break
            iters = iters + 1

        # 根据伪标签更新原型

        score_updata, indices_updata, knn_distances_updata, knn_pred_updata, scores_updata = classifier(support_features2, support_ys2, query_features,
                                                                     support_features, support_ys, 5)
        proto_pred_updata = protoPred(score_updata, support_ys)
        # print("iters:",iters)
        # print("torch.abs(u - P_tensor.sum(dim=1)):", torch.abs(u - P_tensor.sum(dim=1)))

        new_pred4 = P_tensor[n_lsamples+persudo_image.size(0):].argmax(dim=1)
        query_features_pred = query_features
        query_label_pred = new_pred4
        fintune_label = new_pred3
        lp_pred = new_pred3
        # iter_acc.append(torch.tensor((new_pred4 == real_label).float().mean().item()) * 100)

    # print(iter_acc)
    # return new_pred2, new_pred2,new_pred3,new_pred4,P_tensor[n_lsamples+persudo_image.size(0):]


    return new_pred2, new_pred3, new_pred4, torch.from_numpy(probs[n_lsamples + persudo_image.size(0):]).cuda()

def remaining_labels(opt, selected_samples):
    #print(opt.no_samples)
    for i in range(len(opt.no_samples)):
        occurrences = np.count_nonzero(selected_samples == i)
        opt.no_samples[i] = opt.no_samples[i] - occurrences
        #opt.no_samples[opt.no_samples<0] = 0
    #print(opt.no_samples)


def im2features(X,Y, model):
    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=8, num_workers=2, pin_memory=False)
    tensor_list = []
    for batch_ndx, sample in enumerate(loader):
        x, _ = sample
        x = x.cuda()
        feat_support, _ = model(x)
        support_features = feat_support
        tensor_list.append(support_features.detach())
        torch.cuda.empty_cache()
    features = torch.cat(tensor_list, 0)
    return features


def pt_map_preprocess(support, query, beta):
    #X = torch.unsqueeze(torch.cat((torch.Tensor(support), torch.Tensor(query)), dim=0), dim=0)
    X = torch.unsqueeze(torch.cat((support, query), dim=0), dim=0)
    #X = scaleEachUnitaryDatas(X)
    #X = centerDatas(X)
    #nve_idx = np.where(X<0)
    #X[nve_idx] *=-1
    X = PT(X, beta)
    X = scaleEachUnitaryDatas(X)
    X = centerDatas(X)
    X = torch.squeeze(X)
    return X[:support.shape[0]], X[support.shape[0]:]

def check_chosen_labels(indices, ys_pred, y):
    correct = 0
    for i in range(indices.shape[0]):
        if ys_pred[i] == y[indices[i]]:
            correct = correct + 1
    return correct

#helper functions from PT-MAP

def centerDatas(datas):
    #PT code
#    datas[:, :] = datas[:, :, :] - datas[:, :].mean(1, keepdim=True)
 #   datas[:, :] = datas[:, :, :] / torch.norm(datas[:, :, :], 2, 2)[:, :, None]
    # centre of mass of all data support + querries
    datas[:, :] -=datas[:, :].mean(1, keepdim=True)# datas[:, :, :] -
    norma = torch.norm(datas[:, :, :], 2, 2)[:, :, None].detach()
    datas[:, :, :] /= norma

    return datas


def scaleEachUnitaryDatas(datas):
    # print(datas.shape)
    norms = datas.norm(dim=2, keepdim=True)
    return datas / norms


def QRreduction(datas):
    ndatas = torch.qr(datas.permute(0,2,1)).R
    ndatas = ndatas.permute(0,2,1)
    return ndatas

def PT(datas, beta):
    datas[:, ] = torch.pow(datas[:, ] + 1e-6, beta)
    return datas

def preprocess_e2e(X, beta, params):
    X = torch.unsqueeze(X, dim=0)
    X = PT(X, beta)
    X = scaleEachUnitaryDatas(X)
    X = centerDatas(X)
    X = torch.squeeze(X)
    return X

class WrappedModel(nn.Module):
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module
    def forward(self, x):
        return self.module(x)

def load_pt_pretrained(params, model):
    model = model.to(params.device)
    cudnn.benchmark = True
    modelfile = get_resume_file(params.checkpoint_dir)
    checkpoint = torch.load(modelfile)
    state = checkpoint['state']
    if params.dataset == 'tieredImagenet':
        state.pop('module.classifier.L.weight_v', None)
        state.pop('module.classifier.L.weight_g', None)
    state_keys = list(state.keys())

    if params.wrap_flag == 0:
        callwrap = False
        #params.wrap_flag = 1
        if 'module' in state_keys[0]:
            callwrap = True
            params.wrap_flag = 1
        if callwrap:
            model = WrappedModel(model)
    model_dict_load = model.state_dict()
    model_dict_load.update(state)
    model.load_state_dict(model_dict_load)
    return model

def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None

    filelist =  [ x  for x in filelist if os.path.basename(x) != 'best.tar' ]
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file

def load_ici_pretrained(params, model):
    filename = "ici_models/"
    if params.dataset == 'CUB':
        filename = filename + "res12_cub.pth.tar"
    elif params.dataset == "tieredImagenet":
        filename = filename + "res12_tiered.pth.tar"
    elif params.dataset == 'miniImagenet':
        filename = filename + "res12_mini.pth.tar"
    else:
        filename = filename +"res12_cifar.pth.tar"
    state_dict = torch.load(filename)
    model.load_state_dict(state_dict)
    model.to(params.device)
    model.eval()
    return model

def dim_reduce(params, support, query):
    no_sup = support.shape[0]
    X = np.concatenate((support, query), axis = 0)
    X = normalize(X)
    if params.reduce == 'isomap':
        from sklearn.manifold import Isomap
        embed = Isomap(n_components=params.d)
    elif params.reduce == 'itsa':
        from sklearn.manifold import LocallyLinearEmbedding
        embed = LocallyLinearEmbedding(n_components=params.d, n_neighbors=5, method='ltsa')
    elif params.reduce == 'mds':
        from sklearn.manifold import MDS
        embed = MDS(n_components=params.d, metric=False)
    elif params.reduce == 'lle':
        from sklearn.manifold import LocallyLinearEmbedding
        embed = LocallyLinearEmbedding(n_components=params.d, n_neighbors=5, eigen_solver='dense')
    elif params.reduce == 'se':
        from sklearn.manifold import SpectralEmbedding
        embed = SpectralEmbedding(n_components=params.d)
    elif params.reduce == 'pca':
        from sklearn.decomposition import PCA
        embed = PCA(n_components=params.d)
    if params.reduce == 'none':
        X = X
    else:
        X = embed.fit_transform(X)
    return X[:no_sup].astype((np.float32)), X[no_sup:].astype((np.float32))
