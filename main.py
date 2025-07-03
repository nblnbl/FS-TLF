import argparse
from config import get_config
import os   
from logger import create_logger
from data import create_torch_dataloader
from data.dataset_spec import Split
import torch
import numpy as np
import random
import json
from utils import accuracy, AverageMeter, delete_checkpoint, save_checkpoint, load_pretrained, auto_resume_helper, load_checkpoint
import torch
import datetime
from models import get_model
from optimizer import build_optimizer, build_scheduler,build_optimizer2
import time
import math
# from openpyxl import Workbook
from torch.utils.tensorboard import SummaryWriter
from models import models



def setup_seed(seed):
    """
    Fix some seeds.
    """
    # 随机数种子设定
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # CUDA中的一些运算，如对sparse的CUDA张量与dense的CUDA张量调用torch.bmm()，它通常使用不确定性算法。
    # 为了避免这种情况，就要将这个flag设置为True，让它使用确定的实现。
    torch.backends.cudnn.deterministic = True

    # 设置这个flag可以让内置的cuDNN的auto-tuner自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。
    # 但是由于噪声和不同的硬件条件，即使是同一台机器，benchmark都可能会选择不同的算法。为了消除这个随机性，设置为 False
    torch.backends.cudnn.benchmark = False

def parse_option():
    parser = argparse.ArgumentParser('Meta-Dataset Pytorch implementation', add_help=False)
    #parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    # parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file',
    #                     default='./configs/evaluation/res12_PN.yaml')
    # parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file',
    #                     default='./configs/PN_KNN/tieredimagenet_res12_5way-5shot.yaml')
    # parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file',
    #                    default='./configs/PN_KNN/miniImageNet_res12_5way-1shot.yaml')
    parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file',
                       default='./configs/evaluation/CUB_res12_PNKNN_5way-5shot.yaml')
    parser.add_argument("--opts",help="Modify config options by adding 'KEY VALUE' pairs. ",default=None,nargs='+',)

    # easy config modification
    parser.add_argument('--train_batch_size', type=int, help="training batch size for single GPU")
    parser.add_argument('--valid_batch_size', type=int, help="validation batch size for single GPU")
    parser.add_argument('--test_batch_size', type=int, help="test batch size for single GPU")
    parser.add_argument('--output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    #parser.add_argument('--is_train', type=int, choices=[0, 1], help="training or testing")
    parser.add_argument('--is_train', type=int, default=0 ,help="training or testing")
    parser.add_argument('--pretrained', type=str, help="pretrained path")
    # parser.add_argument('--pretrained', type=str, default='./pretrain/mini-resnet12.pth')
    parser.add_argument('--tag', help='tag of experiment',default='test')
    parser.add_argument('--resume', help='resume path')

    parser.add_argument('--k', default=5, help="KNN")
    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config





def test(config):
    # test_dataloader, test_dataset = create_torch_dataloader(Split.TEST, config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")

    model = get_model(config).cuda()

    optimizer = build_optimizer2(config, model)
    # lr_scheduler = build_scheduler(config, optimizer, len(test_dataloader))

    step = 0

    # if config.MODEL.PRETRAINED:
    #     # load_pretrained(config, model, logger)

    # if model has adapters like in TSA
    if hasattr(model, 'mode') and model.mode == "NCC":
        model.append_adapter()

    logger.info("Start testing")

    # with torch.no_grad():
    knn_acc, proto_acc, acc1, loss, ci,label_acc1,label_num1,label_acc2,label_num2,label_acc,label_num,label_num_total,than16_1,than16_2,than16,balance_acc_num1,balance_acc_num2,balance_acc_num3,acc5,acc3 \
        = testing(config, model, model,  model,optimizer,model,step)
    logger.info(
        f"Test Accuracy of {config.DATA.TEST.DATASET_NAMES[0]}: knn_acc:{knn_acc:.2f}%,proto_acc:{proto_acc:.2f}%,acc1:{acc1:.2f}%+-{ci:.2f},acc3:{acc3:.2f}%+-{ci:.2f},acc5:{acc5:.2f}%+-{ci:.2f}")
    logger.info(
        f"stastic : {label_acc1:.2f}%,{label_num1:.2f},{label_acc2:.2f}%,{label_num2:.2f},{label_acc:.2f}%,{label_num:.2f},{label_num_total:.2f}")
    logger.info(
        f"stastic : {than16_1:.2f},{than16_2:.2f},{than16:.2f}")
    logger.info(
        f"stastic_balance : {balance_acc_num1:.2f},{balance_acc_num2:.2f},{balance_acc_num3:.2f}")
    # logging testing results in config.OUTPUT/results.json
    path = os.path.join(config.OUTPUT, "results.json")
    if os.path.exists(path):
        with open(path, 'r') as f:
            result_dic = json.load(f)
    else:
        result_dic = {}

    # by default, we assume there is only one dataset to be tested at a time.
    result_dic[f"{config.DATA.TEST.DATASET_NAMES[0]}"] = [acc1, ci]

    with open(path, 'w') as f:
        json.dump(result_dic, f)




@torch.no_grad()
def testing(config, dataset, data_loader, model,optimizer=None,lr_scheduler=None,step=None):
    model.eval()

    if (config.DATA.TEST.DATASET_NAMES[0] == 'miniImageNet'):
        class_num = 20
        perclass_num = 600
        if (config.DATA.TEST.EPISODE_DESCR_CONFIG.NUM_SUPPORT==1):
            data = torch.load("./feature_aug/mini/1-shot/minifeaturesAS1.pt11", map_location="cuda")[80:]
        elif (config.DATA.TEST.EPISODE_DESCR_CONFIG.NUM_SUPPORT==5):
            data = torch.load("./feature_aug/mini/5-shot/minifeaturesAS1.pt55", map_location="cuda")[80:]
    elif (config.DATA.TEST.DATASET_NAMES[0] == 'tieredimagenet'):
        class_num = 160
        perclass_num = 1300
        if (config.DATA.TEST.EPISODE_DESCR_CONFIG.NUM_SUPPORT == 1):
            data = torch.load("./feature_aug/tiered/1-shot/tieredfeaturesAS1.pt11", map_location="cuda")[448:]
        elif (config.DATA.TEST.EPISODE_DESCR_CONFIG.NUM_SUPPORT == 5):
            data = torch.load("./feature_aug/tiered/5-shot/tieredfeaturesAS1.pt55", map_location="cuda")[448:]
    elif (config.DATA.TEST.DATASET_NAMES[0] == 'Birds'):
        class_num = 30
        perclass_num = 60
        if (config.DATA.TEST.EPISODE_DESCR_CONFIG.NUM_SUPPORT == 1):
            data = torch.load("./feature_aug/CUB/1-shot/cubfsfeaturesAS1.pt11", map_location="cuda")[170:]
        elif (config.DATA.TEST.EPISODE_DESCR_CONFIG.NUM_SUPPORT == 5):
            data = torch.load("./feature_aug/CUB/5-shot/cubfsfeaturesAS1.pt55", map_location="cuda")[170:]
    else:
        perclass_num = 0
        class_num = 0
        datasetTemp = 0

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    acc2_meter = AverageMeter()
    acc3_meter = AverageMeter()
    acc4_meter = AverageMeter()
    acc5_meter = AverageMeter()
    knn_acc_meter = AverageMeter()
    proto_acc_meter = AverageMeter()

    label_acc1_meter = AverageMeter()
    label_num1_meter = 0
    label_acc2_meter = AverageMeter()
    label_num2_meter = 0
    label_acc_meter = AverageMeter()
    label_num_meter = 0
    label_total_num_meter = 0
    than1_meter = 0
    than2_meter = 0
    than_meter = 0
    balance_acc_num1 = 0
    balance_acc_num2 = 0
    balance_acc_num3 = 0



    end = time.time()
    accs = []
    acc5s = []
    n_runs = config.DATA.TEST.EPISODE_DESCR_CONFIG.NUM_TASKS_PER_EPOCH
    n_ways = config.DATA.TEST.EPISODE_DESCR_CONFIG.NUM_WAYS
    n_shot = config.DATA.TEST.EPISODE_DESCR_CONFIG.NUM_SUPPORT
    n_queries = config.DATA.TEST.EPISODE_DESCR_CONFIG.NUM_QUERY

    n_lsamples = n_ways * n_shot
    n_usamples = n_ways * n_queries
    n_samples = n_lsamples + n_usamples

    labels_f = torch.arange(n_ways).view(1, 1, n_ways).expand(n_runs, n_shot + n_queries, 5).clone().view(n_runs,
                                                                                                          n_samples)

    elements_per_class = [perclass_num] * class_num
    # (1000,5)
    run_classes = torch.LongTensor(n_runs, n_ways).to('cuda')
    # (1000,5,20)
    run_indices = torch.LongTensor(n_runs, n_ways, n_shot + n_queries).to('cuda')
    for i1 in range(n_runs):
        # run_classes 内容再20以内
        run_classes[i1] = torch.randperm(class_num)[:n_ways]
        for j1 in range(n_ways):
            # elements_per_class内容[600,600,600,....] 20个600
            run_indices[i1, j1] = torch.randperm(elements_per_class[run_classes[i1, j1]])[:n_shot + n_queries]

    c = torch.zeros(n_samples, 640)
    ndatas2 = torch.zeros(n_runs, n_samples, 640)
    for i in range(n_runs):
        selected_data = []
        for j in range(5):  # 每一轮有5个类别
            class_idx = run_classes[i, j]  # 获取类别索引
            sample_idx = run_indices[i, j]  # 获取该类别下的20个样本索引
            selected_data.append(data[class_idx, sample_idx])  # 从data中选取数据

        for k in range(n_shot + n_queries):
            for k2 in range(5):
                c[(k * 5) + k2] = selected_data[k2][k]
        ndatas2[i] = c

    # ndatas2.set_epoch()
    for idx, batches in enumerate(ndatas2):
        # dataset_index, imgs, labels = batches

        l_batches = labels_f[idx]

        # 分割张量
        support = batches[:n_lsamples]  # 前25个
        query = batches[n_lsamples:n_samples]  # 剩下的75个

        l_support = l_batches[:n_lsamples]  # 前25个
        l_query = l_batches[n_lsamples:n_samples]  # 剩下的75个

        # data_dict = {
        #     'support': support.unsqueeze(0),
        #     'query': query.unsqueeze(0)
        # }
        #
        # l_data_dict = {
        #     'support': l_support.unsqueeze(0),
        #     'query': l_query.unsqueeze(0)
        # }

        data_dict = {
            'support': support,
            'query': query
        }

        l_data_dict = {
            'support': l_support,
            'query': l_query
        }

        imgs = [data_dict]
        labels = [l_data_dict]

        dataset_index = 1
        loss, knn_acc, proto_acc, acc,label_acc1,label_num1,label_acc2,label_num2,label_acc,label_num,label_total_num,than16_1,than16_2,than16,balance_acc1,balance_acc2,balance_acc3,acc2,acc3,acc4,acc5= model.test_forward(imgs, labels, dataset_index,model = model,optimizer = optimizer,step = step)

        # if config.TRAIN.SCHEDULE_PER_STEP:
        #     lr_scheduler.step_update(step)
        #     step += 1
        optimizer.zero_grad()

        accs.extend(acc)
        acc5s.extend(acc5)
        acc = torch.mean(torch.stack(acc))
        acc2 = torch.mean(torch.stack(acc2))
        acc3 = torch.mean(torch.stack(acc3))
        acc4 = torch.mean(torch.stack(acc4))
        acc5 = torch.mean(torch.stack(acc5))

        if label_acc1:
            label_acc1 = torch.mean(torch.stack(label_acc1))
        else:
            label_acc1 = torch.zeros(1)

        knn_acc = torch.mean(torch.stack(knn_acc))
        proto_acc = torch.mean(torch.stack(proto_acc))

        # label_acc1 = torch.mean(torch.stack(label_acc1))
        #
        # label_acc2 = torch.mean(torch.stack(label_acc2))
        #
        # label_acc = torch.mean(torch.stack(label_acc))

        loss_meter.update(loss.item())
        acc_meter.update(acc.item())
        acc2_meter.update(acc2.item())
        acc3_meter.update(acc3.item())
        acc4_meter.update(acc4.item())
        acc5_meter.update(acc5.item())
        label_acc1_meter.update(label_acc1.item())
        knn_acc_meter.update(knn_acc.item())
        proto_acc_meter.update(proto_acc.item())
        acc_meter.update(acc.item())

        # label_acc1_meter.update(label_acc1.item())
        # label_num1_meter+=label_num1
        # label_acc2_meter.update(label_acc2.item())
        #
        # label_num2_meter += label_num2
        # label_acc_meter.update(label_acc.item())

        label_num_meter += label_num
        label_total_num_meter += label_total_num
        than1_meter+=than16_1
        than2_meter += than16_2
        than_meter += than16
        balance_acc_num1 += balance_acc1
        balance_acc_num2 += balance_acc2
        balance_acc_num3 += balance_acc3

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            logger.info(
                f'Test: [{idx + 1}/{len(ndatas2)}]\t'
                f'Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                f'Loss {loss_meter.val:.2f} ({loss_meter.avg:.2f})\t'
                f'knn_Acc {knn_acc_meter.val:.2f} ({knn_acc_meter.avg:.2f})\t'
                f'proto_Acc {proto_acc_meter.val:.2f} ({proto_acc_meter.avg:.2f})\t'
                f'Acc@1 {acc_meter.val:.2f} ({acc_meter.avg:.2f})\t'
                f'Acc@2 {acc2_meter.val:.2f} ({acc2_meter.avg:.2f})\t'
                f'Acc@3 {acc3_meter.val:.2f} ({acc3_meter.avg:.2f})\t'
                f'Acc@4 {acc4_meter.val:.2f} ({acc4_meter.avg:.2f})\t'
                f'Acc@5 {acc5_meter.val:.2f} ({acc5_meter.avg:.2f})\t'
                f'label_acc1_meter {label_acc1_meter.val:.2f} ({label_acc1_meter.avg:.2f})\t'
            )
    accs = torch.stack(accs)
    acc5s = torch.stack(acc5s)
    ci = (1.96 * torch.std(acc5s) / math.sqrt(acc5s.shape[0])).item()

    return knn_acc_meter.avg, proto_acc_meter.avg, acc_meter.avg, loss_meter.avg, ci,label_acc1_meter.avg,label_num1_meter,\
           label_acc2_meter.avg,label_num2_meter,label_acc_meter.avg,label_num_meter,label_total_num_meter,than1_meter,\
           than2_meter,than_meter,balance_acc_num1,balance_acc_num2,balance_acc_num3,acc5_meter.avg,acc3_meter.avg


if __name__ == '__main__':


    args, config = parse_option()
    # torch.cuda.set_device(config.GPU_ID)

    config.defrost()

    config.freeze()

    setup_seed(config.SEED)

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}")

    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")

    if config.IS_TRAIN:
        train(config)
    else:
        test(config)

