#Adapted from Swintransformer
import collections
import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
# from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]

def protoPred(output, target, topk=(1,)):

    maxk = min(max(topk), output.size()[1])

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()

    return pred


#如果原型和knn预测一样，并且k近邻中大于三个
def addImages(proto_pred,knn_pred,indices,knn_train_lable):
    # 如果 indices 是 numpy.ndarray，将其转换为 Tensor 并放到与 knn_train_lable 相同的设备上
    if isinstance(indices, np.ndarray):
        indices = torch.tensor(indices, device=knn_train_lable.device)

    # print(f"knn_train_lable 形状: {knn_train_lable.shape}, 设备: {knn_train_lable.device}, 数据类型: {knn_train_lable.dtype}")
    # print(f"indices 形状: {indices.shape}, 设备: {indices.device}, 数据类型: {indices.dtype}")
    # 确保 indices 和 knn_train_lable 在同一设备上
    if indices.device != knn_train_lable.device:
        indices = indices.to(knn_train_lable.device)
    proto_pred = proto_pred[0]
    # 计算标签差异值并求绝对值
    # print(np.amax(indices))
    # print(indices)
    # print(knn_train_lable)
    #
    # if np.amax(indices)==51:
    #     print("------")
    # 检查是否有越界索引
    # max_index = knn_train_lable.size(0)
    # out_of_bounds = (indices >= max_index).any()
    #
    # if out_of_bounds:
    #     print("存在越界索引值！")
    # else:
    #     print("所有索引值都在有效范围内。")
    indices = indices.long()
    label_diff = torch.abs(knn_train_lable[indices] - knn_pred.unsqueeze(1).expand(-1, indices.shape[1]))
    # 计算零元素的数量
    zero_count = torch.sum(label_diff == 0, dim=1)

    addImage = []
    addLabel = []
    for i,j in enumerate(proto_pred):
        if (proto_pred[i]==knn_pred[i] and zero_count[i]>4):
            addImage.append(i)
            addLabel.append(j)

    return addImage,addLabel


def addImages3(proto_pred,knn_pred,indices,knn_train_lable,lp_pred):
    proto_pred = proto_pred[0]

    if isinstance(indices, np.ndarray):
        indices = torch.tensor(indices, dtype=torch.long, device=knn_train_lable.device)
    # 计算标签差异值并求绝对值
    label_diff = torch.abs(knn_train_lable[indices] - knn_pred.unsqueeze(1).expand(-1, indices.shape[1]))
    # 计算零元素的数量
    zero_count = torch.sum(label_diff == 0, dim=1)

    addImage = []
    addLabel = []
    for i,j in enumerate(proto_pred):
        # if (proto_pred[i].item()==lp_pred[i].item() and lp_pred[i].item()==knn_pred[i] and zero_count[i]>4):
        if (proto_pred[i].item()==lp_pred[i].item()):
            addImage.append(i)
            addLabel.append(j)

    return addImage,addLabel
#如果原型和knn预测一样，并且k近邻中大于三个
def addImages2(proto_pred,knn_pred,indices,knn_train_lable):
    proto_pred = proto_pred[0]

    # 计算标签差异值并求绝对值
    label_diff = torch.abs(knn_train_lable[indices] - knn_pred.unsqueeze(1).expand(-1, indices.shape[1]))
    # 计算零元素的数量
    zero_count = torch.sum(label_diff == 0, dim=1)

    addImage = []
    addLabel = []
    for i,j in enumerate(proto_pred):
        if (proto_pred[i]==knn_pred[i] and zero_count[i]>3):
            addImage.append(i)
            addLabel.append(j)

    return addImage,addLabel
def calculate_accuracy(predictions, true_labels):
    # 确保预测标签和真实标签的长度相同
    assert len(predictions) == len(true_labels), "预测标签和真实标签的长度不一致"

    # 计算正确预测的数量
    correct_predictions = sum(p == t for p, t in zip(predictions, true_labels))

    # 计算准确率
    accuracy = correct_predictions / len(true_labels)

    return accuracy


# def addImages2(support_images,support_label,query_images,labelnum,temp):
#     if support_images.dim() == 4:
#         support_images = F.adaptive_avg_pool2d(support_images, 1).squeeze_(-1).squeeze_(-1)
#         query_images = F.adaptive_avg_pool2d(query_images, 1).squeeze_(-1).squeeze_(-1)
#
#     assert support_images.dim() == query_images.dim() == 2
#     support_images = F.normalize(support_images, p=2, dim=1, eps=1e-12)
#     query_images = F.normalize(query_images, p=2, dim=1, eps=1e-12)
#
#     # 支持集和测试集连起来
#     knn_support_images = torch.cat((support_images, query_images), dim=0)
#
#     nearest_neighbors = NearestNeighbors(n_neighbors=4, metric='cosine').fit(knn_support_images.cpu().detach().numpy())
#     # 支持集的最近邻
#     distances, indices = nearest_neighbors.kneighbors(support_images.cpu().detach().numpy())
#
#     query_label = [-1 for _ in range(labelnum)]
#
#     indices2 = indices[:, 1:]
#     #-1 代表没有标签，-2代表有一个以上标签，大于-1代表样本标签
#     for i ,j in enumerate(indices2):
#
#         for j1 in j:
#             # 如果k近邻中有查询集
#             if j1>=temp:
#                 j2 = j1-temp
#                 if(query_label[j2]<-1):
#                     continue
#                 elif(query_label[j2]>-1 and query_label[j2]!=support_label[i]):
#                     query_label[j2]=-2
#                 elif (query_label[j2] > -1 and query_label[j2] == support_label[i]):
#                     continue
#                 else:
#                     query_label[j2]=support_label[i].item()
#
#     return query_label

def isThan16(label):

    num1 = 0

    # 伪标签中每个类有多少个
    count_list = [0] * 5

    # 遍历列表，统计每个数字的出现次数

    for num in label:
        if num>-1 and num<5:
            count_list[num] += 1
        if num>4:
            print(label)


    for j1, j2 in enumerate(count_list):
        if j2 > 16:
            num1 += 1

    return num1

# 统计改变前后的正确数量
def stastic_balance(idx1,idx2,real_idx):

    sum = len(idx1)
    idx1_acc = 0
    idx2_acc = 0



    idx1_acc = torch.sum(idx1 == real_idx)
    idx2_acc = torch.sum(idx2 == real_idx)

    return idx1_acc,idx2_acc,sum
# 解决类别不平衡
def class_balance(proto_score,knn_indices,knn_trainLabel,queryImages_idx,queryLabel,k):

    temp_num = 16

    images_idx = queryImages_idx
    label = queryLabel
    updata_idx = []
    #伪标签中每个类有多少个
    count_list = [0] * 5

    # 遍历列表，统计每个数字的出现次数
    for num in label:
        count_list[num] += 1

    # 统计大于16的类别总数和有哪些类别
    greater_than_5 = 0
    greater_than_5_idx = []
    for j1,j2 in enumerate(count_list) :
        if j2>temp_num:
            greater_than_5+=1
            greater_than_5_idx.append(j1)

    has_greater_than_16 = any(count > temp_num for count in count_list)
    proto_confi = torch.zeros(75, 5)
    knnConfi = torch.zeros(75, 5)

    if has_greater_than_16:


        logits2 = torch.exp(proto_score)
        # 计算每个样本的总和
        sum_logits = torch.sum(logits2, dim=1)

        proto_confi = logits2/sum_logits.reshape(75,1)


        numLabel0 = torch.sum(knn_trainLabel[knn_indices] == 0, dim=1)
        numLabel1 = torch.sum(knn_trainLabel[knn_indices] == 1, dim=1)
        numLabel2 = torch.sum(knn_trainLabel[knn_indices] == 2, dim=1)
        numLabel3 = torch.sum(knn_trainLabel[knn_indices] == 3, dim=1)
        numLabel4 = torch.sum(knn_trainLabel[knn_indices] == 4, dim=1)

        knnConfi0 = numLabel0 / k
        knnConfi1 = numLabel1 / k
        knnConfi2 = numLabel2 / k
        knnConfi3 = numLabel3 / k
        knnConfi4 = numLabel4 / k

        knnConfi0 = knnConfi0.reshape(75,1)
        knnConfi1 = knnConfi1.reshape(75, 1)
        knnConfi2 = knnConfi2.reshape(75, 1)
        knnConfi3 = knnConfi3.reshape(75, 1)
        knnConfi4 = knnConfi4.reshape(75, 1)

        #knnConfi = [[knnConfi0[i1], knnConfi1[i1], knnConfi2[i1], knnConfi3[i1], knnConfi4[i1]] for i1 in range(len(knnConfi0))]
        knnConfi = torch.cat((knnConfi0, knnConfi1, knnConfi2, knnConfi3, knnConfi4), dim=1)
        # 分析knn置信度是不是能决定最终置信度的大小
        confi = proto_confi+0.1*knnConfi



        label_self_index = [-1]*75

        for labelidx1,labelidx2 in enumerate(images_idx):
            label_self_index[labelidx2] = labelidx1

        for than_5_idx in greater_than_5_idx:
            confi_temp = {}
            for images_idx1,images_idx2 in enumerate(images_idx):
                if label[images_idx1] ==than_5_idx:
                    confi_temp[images_idx2] = confi[images_idx2][than_5_idx].item()

            sorted_dict = dict(sorted(confi_temp.items(), key=lambda item: item[1], reverse=True))

            keys = list(sorted_dict.keys())

            #多出来的那几个伪标签的索引
            add_images_idx = []
            for keys_idx in range(len(keys)):
                if keys_idx>temp_num-1:
                    add_images_idx.append(keys[keys_idx])
                    updata_idx.append(keys[keys_idx])

            for x in add_images_idx:
                confi_line = confi[x]
                sorted_indices = sorted(range(len(confi_line)), key=lambda x: confi_line[x], reverse=True)

                for x1 in sorted_indices:
                    if(count_list[x1]<temp_num):

                        label[label_self_index[x]] = x1
                        # print("***********")
                        count_list[than_5_idx]-=1
                        count_list[x1]+=1
                        break
    # 伪标签中每个类有多少个
    count_list2 = [0] * 5

    # 遍历列表，统计每个数字的出现次数
    for num in label:
        count_list2[num] += 1

    # 统计大于16的类别总数和有哪些类别
    greater_than_51 = 0

    for j1, j2 in enumerate(count_list2):
        if j2 > temp_num:
            greater_than_51 += 1
    if greater_than_51>0:
        print("-----------------------------------------------------------")
    return images_idx,label,updata_idx,proto_confi,knnConfi


def getprotoconfi(logits2,label):
    logits2 = torch.exp(logits2)
    # 计算每个样本的总和
    sum_logits = torch.sum(logits2, dim=1)

    # 通过索引获取每个样本对应类别的logit
    selected_logits = logits2[range(logits2.size(0)), label]



    # 计算x3
    confi1 = selected_logits / sum_logits

    # 归一化
    confisum2 = torch.sum(confi1)
    newconfi2 = confi1 / confisum2


    return confi1.cuda(),newconfi2.cuda()


# def knn_st(logits,knn_test_lable,indices,x1,x2):
#
#
#     count_tensor = torch.zeros(x1, x2, dtype=torch.int, device='cuda')
#     for i in range(x2):
#         count_tensor[:, i] = (knn_test_lable[indices] == i).sum(dim=1)
#
#     max_indices = []
#     for i in range(x1):
#         row = count_tensor[i]
#         max_value = torch.max(row).item()
#         indices = (row == max_value).nonzero()  # 找到与最大值相等的索引
#         max_indices.append(indices)
#
#
#     element_counts = []
#
#     for tensor in max_indices:
#         # 使用size函数获取每个张量的维度信息，然后取第一个维度的大小
#         num_elements = tensor.size(0)
#         element_counts.append(num_elements)
#
#
#     res = torch.empty(0, dtype=torch.int, device='cuda')
#
#     num=0
#     for i, j in enumerate(element_counts, 0):
#         if(j>1):
#             #max_indices[i]
#             min_values, min_indices = torch.max(logits[i][max_indices[i]], dim=0)
#
#             res = torch.cat((res, max_indices[i][min_indices].to(torch.int)))
#             num+=1
#         else:
#             res = torch.cat((res, max_indices[i].to(torch.int)))
#
#     return res.t()[0]

def knn_st(knn_distances,knn_test_lable,indices,x1,x2,k):

    disSum = torch.zeros(x1, x2)

    for i in range(x1):
        for j in range(k):
            label = (knn_test_lable[indices])[i, j]  # 获取标签值
            distance = knn_distances[i, j]  # 获取距离值
            disSum[i, label] += distance  # 将距离值累加到对应标签的位置


    count_tensor = torch.zeros(x1, x2, dtype=torch.int, device='cuda')
    for i in range(x2):
        count_tensor[:, i] = (knn_test_lable[indices] == i).sum(dim=1)


    max_indices = []
    for i in range(x1):
        row = count_tensor[i]
        max_value = torch.max(row).item()
        indices2 = (row == max_value).nonzero()  # 找到与最大值相等的索引
        max_indices.append(indices2)


    element_counts = []

    for tensor in max_indices:
        # 使用size函数获取每个张量的维度信息，然后取第一个维度的大小
        num_elements = tensor.size(0)
        element_counts.append(num_elements)


    res = torch.empty(0, dtype=torch.int, device='cuda')



    num=0
    for i, j in enumerate(element_counts, 0):
        if(j>1):
            #max_indices[i]
            min_values, min_indices = torch.max(disSum[i][max_indices[i]], dim=0)

            res = torch.cat((res, max_indices[i][min_indices].to(torch.int)))
            num+=1
        else:
            res = torch.cat((res, max_indices[i].to(torch.int)))

    return res.t()[0]




def getknnconfi(indices,knn_train_lable,knn_test_lable,k):
    # 计算标签差异值并求绝对值
    label_diff = torch.abs(knn_train_lable[indices] - knn_test_lable.unsqueeze(1).expand(-1, indices.shape[1]))
    # 计算零元素的数量
    non_zero_count = torch.sum(label_diff == 0, dim=1)
    # print('0的个数：',(non_zero_count==0).sum(),'1的个数：',(non_zero_count==1).sum(),'2的个数：',(non_zero_count==2).sum(),'3的个数：',(non_zero_count==3).sum())

    # 计算损失并进行指数运算
    newconfi = (non_zero_count.float() / k)
    confisum2 = newconfi.sum()
    # 计算平均损失
    newconfi2 = newconfi/confisum2

    return newconfi.cuda(),newconfi2.cuda()



def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger, topK, step):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config,
                  'step':step}
    if topK is not None:
        save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}_top{topK}.pth')
    else:
        save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")

def delete_checkpoint(config, topK=None, epoch = None):
    if topK is not None:
        for file_ in os.listdir(config.OUTPUT):
            # delete the topK checkpoint
            if f"top{config.SAVE_TOP_K_MODEL}" in file_:
                os.remove(os.path.join(config.OUTPUT, file_))
                break
        for j in range(config.SAVE_TOP_K_MODEL-1,topK-1, -1):
            # move the checkpoints 
            for file_ in os.listdir(config.OUTPUT):
                if f"top{j}" in file_:
                    os.rename(os.path.join(config.OUTPUT, file_),
                        os.path.join(config.OUTPUT, file_).replace(f"top{j}", f"top{j+1}"))
                    break
    elif epoch is not None:
        if os.path.exists(os.path.join(config.OUTPUT, f"ckpt_epoch_{epoch}.pth")):
            os.remove(os.path.join(config.OUTPUT, f"ckpt_epoch_{epoch}.pth"))
        
def load_pretrained(config, model, logger):
    logger.info(f"==============> Loading weight {config.MODEL.PRETRAINED} for fine-tuning......")
    checkpoint = torch.load(config.MODEL.PRETRAINED, map_location='cpu')

    possible_keys = ["state_dict", "model", "models"]

    flag = True
    for key in possible_keys:
        if key in checkpoint.keys():
            the_key = key
            flag = False
            break
    if flag:
        state_dict = checkpoint
    else:    
        state_dict = checkpoint[the_key]
    
    state_keys = list(state_dict.keys())
    for i, key in enumerate(state_keys):
        if "backbone" in key:
            newkey = key.replace("backbone.", "")
            state_dict[newkey] = state_dict.pop(key)
        if "classifier" in key:
            state_dict.pop(key)
    
    msg = model.backbone.load_state_dict(state_dict,strict=False)
    logger.warning(msg)

    logger.info(f"=> loaded successfully '{config.MODEL.PRETRAINED}'")

    del checkpoint
    torch.cuda.empty_cache()

def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file

def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')

    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    max_accuracy = [0.0]*config.SAVE_TOP_K_MODEL
    step = 0
    if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
    if 'max_accuracy' in checkpoint:
        max_accuracy = checkpoint['max_accuracy']
        logger.info(f"load max_accuracy:{max_accuracy}")
    if 'step' in checkpoint:
        step = checkpoint['step']
        logger.info(f"load step:{step}")

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy, step
# 统计查询样本到原型的相似度，以及查询点距离k个近邻相似度
def statistic2(knnPred,protoPred,label,scores,knn_distances,knn_distances_label,knn_confi_pre,knn_confi,proto_confi_pre,proto_confi,addImage,addLabel):

    array1 = scores.cpu().numpy()
    array2 = protoPred[0].cpu().numpy()
    array3 = label.cpu().numpy()
    array4 = knn_distances
    array5 = knnPred.cpu().numpy()
    array6 = knn_distances_label.cpu().numpy()
    array7 = knn_confi_pre.cpu().numpy()
    array8 = knn_confi.cpu().numpy()
    array9 = proto_confi_pre[0].cpu().numpy()
    array10 = proto_confi[0].cpu().numpy()

    array2 = array2[:, np.newaxis]  # 变成75*1
    array3 = array3[:, np.newaxis]  # 变成75*1
    array5 = array5[:, np.newaxis]  # 变成75*1
    array7 = array7[:, np.newaxis]
    array8 = array8[:, np.newaxis]
    array9 = array9[:, np.newaxis]
    array10 = array10[:, np.newaxis]
    combined_array = np.hstack((array1, array9,array10,array2, array3,array4,array6,array7,array8,array5))

    different_indices = np.where(array2 != array5)[0]

    knn_error_num = 0
    for i1 ,dif_idx in enumerate(different_indices):
        if(array5[dif_idx][0]!=array3[dif_idx][0]):
            knn_error_num+=1

    proto_error_num = 0
    for i2, dif_idx in enumerate(different_indices):
        if (array2[dif_idx][0] != array3[dif_idx][0]):
            proto_error_num += 1
    knn_dif_label = []
    proto_dif_label = []
    correct_dif_label = []

    for i3, dif_idx in enumerate(different_indices):
        knn_dif_label.append(array5[dif_idx][0])
        proto_dif_label.append(array2[dif_idx][0])
        correct_dif_label.append(array3[dif_idx][0])




    print("knn和原型不一致的数量：",len(different_indices))
    print("knn对的数量：", len(different_indices)-knn_error_num)
    print("原型对的数量：", len(different_indices)-proto_error_num)


    print("不一致中knn预测结果：", knn_dif_label)
    print("不一致中原型预测结果：", proto_dif_label)
    print("不一致中正确预测结果：", correct_dif_label)
    print(addImage)
    print(addLabel)

    df = pd.DataFrame(combined_array)

    # 将DataFrame写入Excel表格
    excel_file = 'statistic2.xlsx'
    df.to_excel(excel_file, index=False)
    print(f"Combined tensor has been written to {excel_file}")

def statistic(knnPred,knnStPred,protoPred,pred,label,knnConfi,protoConfi,pre_knn_confi,pre_proto_confi):
    errors_knn = torch.abs(knnPred - label)
    errors_knnst = torch.abs(knnStPred - label)
    errors_proto = torch.abs(protoPred - label)
    errors_pred = torch.abs(pred - label)

    result_knn = torch.where(errors_knn == 0, 0, 1)
    result_knnst = torch.where(errors_knnst == 0, 0, 1)
    result_proto = torch.where(errors_proto == 0, 0, 1)
    result_pred = torch.where(errors_pred == 0, 0, 1)

    errorsSum_knn = torch.sum(result_knn == 1)
    errorsSum_knnst = torch.sum(result_knnst == 1)
    errorsSum_proto = torch.sum(result_proto == 1)
    errorsSum_pred = torch.sum(result_pred == 1)

    # 创建一个空字典，用于统计各个数的数量
    count_dict = {}

    # 遍历数据，统计每个数出现的次数
    for num in pre_knn_confi:
        if ("{:.4f}".format(num.item())) in count_dict:
            count_dict[("{:.4f}".format(num.item()))] += 1
        else:
            count_dict[("{:.4f}".format(num.item()))] = 1

    print(count_dict)

    sorted_data = sorted(pre_proto_confi[0])
    n = len(sorted_data)
    if n % 2 == 0:
        median_value = (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
    else:
        median_value = sorted_data[n // 2]
    print("原型置信度的最大值为：",max(pre_proto_confi[0]).item())
    print("原型置信度的最小值为：",min(pre_proto_confi[0]).item())
    print("原型置信度的平均值为：",sum(pre_proto_confi[0]) / len(pre_proto_confi[0]))
    print("原型置信度的中值为：",median_value.item())

    print("knn预测错误数量为：",errorsSum_knn.item())
    print("knnst预测错误数量为：",errorsSum_knnst.item())
    print("proto预测错误数量为：" , errorsSum_proto.item() )
    print("pred预测错误数量为：" , errorsSum_pred.item())

    proto_knn = torch.where(protoConfi > knnConfi, 1, 0)
    proto_confi_sum = torch.sum(proto_knn == 1)

    proto_selected_predicted = [protoPred[0][i].item() for i in range(len(proto_knn[0])) if proto_knn[0][i] == 1]

    proto_selected_predicted2 = [knnPred[i].item() for i in range(len(proto_knn[0])) if proto_knn[0][i] == 1]
    # proto_selected_predicted = []
    # for i in range(len(proto_knn[0])):
    #     if proto_knn[0][i].item() == 1:
    #         proto_selected_predicted.append(protoPred[0][i].item())


    proto_selected_true_labels = [label[i].item() for i in range(len(proto_knn[0])) if proto_knn[0][i] == 1]

    acc1 = sum(1 for elem1, elem2 in zip(proto_selected_predicted, proto_selected_true_labels) if elem1 != elem2)

    acc11 = sum(1 for elem1, elem2 in zip(proto_selected_predicted2, proto_selected_true_labels) if elem1 != elem2)
    print("原型比knn置信度大的数量为：",proto_confi_sum)
    print("其中预测错误的数量为：" , acc1)
    print("原型预测的这些knn预测错误数量：", acc11)

    knn_proto = torch.where(knnConfi > protoConfi, 1, 0)
    knn_confi_sum = torch.sum(knn_proto == 1)

    knn_selected_predicted = [knnPred[i].item() for i in range(len(knn_proto[0])) if knn_proto[0][i].item() == 1]
    knn_selected_predicted2 = [protoPred[0][i].item() for i in range(len(knn_proto[0])) if knn_proto[0][i].item() == 1]
    knn_selected_true_labels = [label[i].item() for i in range(len(knn_proto[0])) if knn_proto[0][i].item() == 1]

    acc2 = sum(1 for elem1, elem2 in zip(knn_selected_predicted, knn_selected_true_labels) if elem1 != elem2)

    acc22 = sum(1 for elem1, elem2 in zip(knn_selected_predicted2, knn_selected_true_labels) if elem1 != elem2)

    print("knn比原型置信度大的数量为：" , knn_confi_sum)
    print("其中预测错误的数量为：" , acc2 )
    print("knn预测的这些原型预测错误个数：", acc22)

    return 0


# 统计查询样本到原型的相似度，以及查询点距离k个近邻相似度
def statistic_balance(score,indices,pred1,pred2,realLabel,proto_confi_stas,knnConfi_stas):

    array1 = score.cpu().numpy()
    array2 = indices
    array6 = proto_confi_stas.cpu().numpy()
    array7 = knnConfi_stas.cpu().numpy()
    array3 = pred1.cpu().numpy()
    array4 = pred2.cpu().numpy()
    array5 = realLabel.cpu().numpy()
    array3 = array3[:, np.newaxis]
    array4 = array4[:, np.newaxis]
    array5 = array5[:, np.newaxis]
    combined_array = np.hstack((array1, array2,array6,array7,array3,array4, array5))


    df = pd.DataFrame(combined_array)

    # 将DataFrame写入Excel表格
    excel_file = 'statistic_balance.xlsx'
    df.to_excel(excel_file, index=False)
    print(f"Combined tensor has been written to {excel_file}")