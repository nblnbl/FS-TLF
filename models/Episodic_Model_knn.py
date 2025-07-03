"""
Module for episodic (meta) training/testing
"""
import numpy as np
import pandas as pd
from architectures import get_backbone, get_classifier,get_classifier2
import torch.nn as nn
import torch.nn.functional as F
from utils import accuracy
from utils import getprotoconfi,getknnconfi,protoPred,knn_st,statistic,statistic2,addImages,addImages2,class_balance,calculate_accuracy,isThan16,stastic_balance,statistic_balance
import torch
import iterative_graph_functions as igf


def calculate_inter_class_distance(vectors, labels):
    """
    计算类间距离（类中心之间的欧氏距离）
    :param vectors: 输入的向量 Tensor，形状为 (n_samples, n_features)
    :param labels: 对应的标签 Tensor，形状为 (n_samples,)
    :return: 类间距离矩阵
    """
    unique_labels = torch.unique(labels)
    num_classes = len(unique_labels)
    class_centers = []

    # 计算每个类别的中心
    for label in unique_labels:
        class_vectors = vectors[labels == label]
        center = torch.mean(class_vectors, dim=0)
        class_centers.append(center)

    class_centers = torch.stack(class_centers)

    # 计算类间距离矩阵
    inter_class_distances = torch.cdist(class_centers, class_centers)

    return inter_class_distances


def calculate_sse(vectors, labels):
    """
    计算平方误差和（SSE）
    :param vectors: 输入的向量 Tensor，形状为 (n_samples, n_features)
    :param labels: 对应的标签 Tensor，形状为 (n_samples,)
    :return: SSE 值
    """
    unique_labels = torch.unique(labels)
    sse = 0

    # 计算每个类别的 SSE 并累加
    for label in unique_labels:
        class_vectors = vectors[labels == label]
        center = torch.mean(class_vectors, dim=0)
        distances = torch.norm(class_vectors - center, dim=1)
        sse += torch.sum(distances ** 2)

    return sse
class EpisodicTraining(nn.Module):
    def __init__(self, config,scale_cls: int =1.0):
        super().__init__()
        self.backbone = get_backbone(config.MODEL.BACKBONE, *config.MODEL.BACKBONE_HYPERPARAMETERS)
        self.classifier = get_classifier(config.MODEL.CLASSIFIER, *config.MODEL.CLASSIFIER_PARAMETERS)
        # self.classifier2 = get_classifier2(config.MODEL.CLASSIFIER, *config.MODEL.CLASSIFIER_PARAMETERS)
        self.train_way = config.DATA.TRAIN.EPISODE_DESCR_CONFIG.NUM_WAYS
        self.query = config.DATA.TRAIN.EPISODE_DESCR_CONFIG.NUM_QUERY
        self.scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(scale_cls), requires_grad=True)
        if config.IS_TRAIN == 0:
            self.support = config.DATA.TEST.EPISODE_DESCR_CONFIG.NUM_SUPPORT
        else:

            self.support = config.DATA.TRAIN.EPISODE_DESCR_CONFIG.NUM_SUPPORT
        self.k = config.K


    def forward(self,img_tasks,label_tasks, *args, model, optimizer,step,**kwargs):
        batch_size = len(img_tasks)
        loss = 0.
        loss2 = 0.
        knn_acc = []
        proto_acc = []
        acc = []
        acc2 = []
        acc3 = []
        acc4 = []
        acc5 = []
        label_acc1 = []
        label_num1 = 0
        label_acc2 = []
        label_num2 = 0
        label_acc = []
        label_num = 0
        label_total_num = 0
        than16_1 = 0
        than16_2 = 0
        than16 = 0
        balance_acc1 = 0
        balance_acc2 = 0
        balance_acc3 = 0
        array = []
        for i, img_task in enumerate(img_tasks):
            support_features = img_task["support"].squeeze_().cuda()
            query_features = img_task["query"].squeeze_().cuda()
            score, indices, knn_distances, knn_pred, scores = self.classifier(support_features,label_tasks[i]["support"].squeeze_().cuda(),query_features, support_features,
                                    label_tasks[i]["support"].squeeze_().cuda(), self.k, **kwargs)

            proto_pred = protoPred(score, torch.squeeze(label_tasks[i]["support"]))
            knn_trainLabel = label_tasks[i]["query"].squeeze_().cpu()

            # addImage, addLabel = addImages2(proto_pred, knn_pred, indices, knn_trainLabel.cuda())
            addImage2, addLabel2 = addImages(proto_pred, knn_pred, indices, knn_trainLabel.cuda())
            # 计算标签差异值并求绝对值---------------这行有问题，label_tasks[i]["query"]和原来代码里的不一样
            label_diff = torch.abs(
                knn_trainLabel[indices] - torch.squeeze(label_tasks[i]["query"]).unsqueeze(1).expand(-1,indices.shape[1]))
            # 计算非零元素的数量
            non_zero_count = torch.sum(label_diff != 0, dim=1)
            # 计算损失并进行指数运算
            # knn_loss = torch.exp(non_zero_count.float() / self.k)
            knn_loss = non_zero_count.float() / self.k
            # 计算平均损失
            knn_loss = knn_loss.mean().item()


            loss += (F.cross_entropy(score, label_tasks[i]["query"].squeeze_().cuda())) + (knn_loss)


            proto_pred = protoPred(score, torch.squeeze(label_tasks[i]["query"]))

            pre_proto_confi, proto_confi = getprotoconfi(scores, proto_pred)  # 获得原型网络的置信度
            pre_knn_confi, knn_confi = getknnconfi(indices, knn_trainLabel, knn_pred.cpu(), self.k)  # 获得knn的置信度

            knn_st_pred = knn_st(knn_distances, knn_trainLabel, indices, self.train_way * self.query, self.train_way,self.k)
            new_pred = torch.where(0.7*proto_confi[0] > 0.3*(knn_confi), proto_pred[0], knn_st_pred.to(torch.long))

            # -----------------------------------------------------

            # score3, indices3, knn_distances3, knn_pred3, scores3 = self.classifier(support_features, label_tasks[i][
            #     "support"].squeeze_().cuda(), query_features[addImage], support_features, label_tasks[i]["support"].squeeze_().cuda(),self.k, **kwargs)
            #
            # # loss2 += (self.scale_cls*F.cross_entropy(score3, torch.stack(addLabel)))
            # loss2 += F.cross_entropy(score3, torch.stack(addLabel))
            # --------------------------------------------------------
            query_ys_pred1, query_ys_pred2, query_ys_pred3 ,P_tensor= igf.iter_balanced_trans(
                img_task["support"].squeeze_().cuda(),
                label_tasks[i]["support"].squeeze_().cuda(),
                img_task["query"].squeeze_().cuda(),
                  self.classifier, torch.squeeze(label_tasks[i]["query"]).cuda())

            # 计算每一行的总和
            row_sums = P_tensor.sum(dim=1, keepdim=True)
            # 进行行归一化
            P_tensor = P_tensor / row_sums
            scores_exp = torch.exp(scores)
            row_sums2 = scores_exp.sum(dim=1, keepdim=True)
            P_tensor_scores = scores_exp/row_sums2


        # 2.12------选择最优lambda-----START
            # 5-shot  88.54
            # weight = [0.04,0.041,0.043,0.045,0.047,0.049,0.05,0.051,0.053,0.055,0.057,0.059,0.06]
            # 5-shot  88.55
            weight = np.arange(0.04, 0.06, 0.0005)
            # weight = np.arange(0.54, 0.56, 0.0005)
            query_images = torch.cat((support_features, query_features[addImage2]), dim=0)
            query_label = torch.cat((label_tasks[i]["support"].squeeze_().cuda(), torch.tensor(addLabel2).cuda()), dim=0)

            query_ys_pred111, query_ys_pred222, query_ys_pred333, P_tensor1 = igf.iter_balanced_trans(
                img_task["support"].squeeze_().cuda(),label_tasks[i]["support"].squeeze_().cuda(),query_images,self.classifier, img_task["support"].squeeze_().cuda())

            # 计算每一行的总和
            row_sums1 = P_tensor1.sum(dim=1, keepdim=True)
            # 进行行归一化
            P_tensor1 = P_tensor1 / row_sums1

            score1, indices1, knn_distances1, knn_pred1, scores1 = self.classifier(support_features, label_tasks[i][
                "support"].squeeze_().cuda(), query_images, support_features,label_tasks[i]["support"].squeeze_().cuda(), self.k,**kwargs)

            scores1_exp = torch.exp(scores1)
            row_sums111 = scores1_exp.sum(dim=1, keepdim=True)
            P_tensor_scores1 = scores1_exp / row_sums111


            minsse = 99999
            maxacweight = 0.05 # lambda
            for q in weight:
                P_tensor11 = q * P_tensor1 + (1-q) * P_tensor_scores1
                pred11 = P_tensor11.argmax(dim=1)
                sse = calculate_sse(query_images, pred11)

                if sse<minsse:
                    minsse = sse
                    maxacweight = q

        # 2.12------选择最优lambda-----END
        #     maxacweight = 0.1
            P_tensor2 = maxacweight*P_tensor+(1-maxacweight)*P_tensor_scores
            # print(maxacweight)
            pred = P_tensor2.argmax(dim=1)

            # ------------------------------------------------------
            #statistic2(knn_pred,proto_pred,label_tasks[i]["query"].squeeze_().cuda(),scores,knn_distances,knn_trainLabel[indices],pre_knn_confi,knn_confi,pre_proto_confi,proto_confi,addImage,addLabel)
            acc.append(#KNN + proto
                torch.tensor((new_pred == torch.squeeze(label_tasks[i]["query"]).cuda()).float().mean().item()) * 100)
            acc2.append(torch.tensor((torch.from_numpy(query_ys_pred1).cuda() == torch.squeeze(#proto + LP
                label_tasks[i]["query"]).cuda()).float().mean().item()) * 100)
            acc3.append(torch.tensor(
                (torch.from_numpy(query_ys_pred2).cuda() == torch.squeeze(
                    label_tasks[i]["query"]).cuda()).float().mean().item()) * 100)
            acc4.append(torch.tensor(
                (query_ys_pred3 == torch.squeeze(label_tasks[i]["query"]).cuda()).float().mean().item()) * 100)
            acc5.append(torch.tensor(#KNN + proto + LP
                (pred == torch.squeeze(label_tasks[i]["query"]).cuda()).float().mean().item()) * 100)
            knn_acc.append(#KNN
                torch.tensor((knn_pred == torch.squeeze(label_tasks[i]["query"]).cuda()).float().mean().item()) * 100)
            proto_acc.append(torch.tensor(#proto
                (proto_pred == torch.squeeze(label_tasks[i]["query"]).cuda()).float().mean().item()) * 100)
            label_total_num+=75
            # acc.append(accuracy(score, label_tasks[i]["query"].cuda())[0])
        loss /= batch_size


        return loss, knn_acc, proto_acc, acc,label_acc1,label_num1,label_acc2,label_num2,label_acc,label_num,label_total_num,than16_1,than16_2,than16,balance_acc1,balance_acc2,balance_acc3,acc2,acc3,acc4,acc5




    def train_forward(self, img_tasks,label_tasks, *args, **kwargs):
        # return self(img_tasks, label_tasks, *args, **kwargs)
        return self.forward2(img_tasks, label_tasks, *args, **kwargs)

    def val_forward(self, img_tasks,label_tasks, *args, **kwargs):

        return self.forward3(img_tasks, label_tasks, *args, **kwargs)

    def test_forward(self, img_tasks,label_tasks, *args, model,optimizer,step,**kwargs):
        return self(img_tasks, label_tasks, *args, model=model,optimizer=optimizer,step = step, **kwargs)

def get_model(config):
    return EpisodicTraining(config)