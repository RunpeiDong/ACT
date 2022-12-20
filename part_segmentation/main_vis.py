"""
Author: Yabin
Date: July 2022
Function: save part segmentation results as colorful point cloud, which could be visualized with MeshLab
"""
import argparse
import os
import torch
import datetime
import logging
import sys
import importlib
import shutil
import provider
import numpy as np
import torch.optim as optim
from timm.scheduler import CosineLRScheduler
from pathlib import Path
from tqdm import tqdm
from dataset import PartNormalDataset
import ipdb


'''
Airplane	02691156
Bag	        02773838
Cap	        02954340
Car	        02958343
Chair	    03001627
Earphone	03261776
Guitar	    03467517
Knife	    03624134
Lamp	    03636649
Laptop	    03642806
Motorbike   03790512
Mug	        03797390
Pistol	    03948459
Rocket	    04099429
Skateboard  04225987
Table	    04379243'''

cmap = np.array([[1.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                 [3.12493437e-02, 1.00000000e+00, 1.31250131e-06],
                 [0.00000000e+00, 6.25019688e-02, 1.00000000e+00],
                 [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                 [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                 [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                 [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                 [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                 [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                 [1.00000000e+00, 0.00000000e+00, 9.37500000e-02]])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
               'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
               'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
               'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pt', help='model name')
    parser.add_argument('--optimizer_part', type=str, default='all', help='training all parameters or optimizing the new layers only')
    parser.add_argument('--batch_size', type=int, default=16, help='batch Size during training')
    parser.add_argument('--epoch', default=300, type=int, help='epoch to run')
    parser.add_argument('--warmup_epoch', default=10, type=int, help='warmup epoch')
    parser.add_argument('--learning_rate', default=0.0002, type=float, help='initial learning rate')
    parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
    # parser.add_argument('--optimizer', type=str, default='AdamW', help='Adam or SGD')
    parser.add_argument('--log_dir', type=str, default='./exp', help='log path')
    # parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--npoint', type=int, default=2048, help='point Number')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    # parser.add_argument('--step_size', type=int, default=20, help='decay step for lr decay')
    # parser.add_argument('--lr_decay', type=float, default=0.5, help='decay rate for lr decay')
    parser.add_argument('--ckpts', type=str, default='../best/pretrain/m0.6R_1_pretrain300.pth', help='ckpts')
    parser.add_argument('--root', type=str, default='../data/shapenetcore_partanno_segmentation_benchmark_v0_normal/', help='data root')
    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    # '''HYPER PARAMETER'''
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('part_seg_visual')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = args.root

    # TRAIN_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split='trainval', normal_channel=args.normal)
    # trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    TEST_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split='test', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=1, shuffle=False, num_workers=10)
    # log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    num_classes = 16
    num_part = 50

    '''MODEL LOADING'''
    ckpts_mae = './log/part_seg/pretrain_official/checkpoints/best_model.pth'
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(exp_dir))
    # shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    classifier = MODEL.get_model(num_part).cuda()
    criterion = MODEL.get_loss().cuda()
    classifier.apply(inplace_relu)
    print('# generator parameters:', sum(param.numel() for param in classifier.parameters()))
    classifier.load_model_from_ckpt_withrename(ckpts_mae)
    start_epoch = 0

    '''MODEL LOADING'''
    ckpts_masksurf = './log/part_seg/pretrain_withnormal_loos_w001_gradualw/checkpoints/best_model.pth'
    MODEL2 = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(exp_dir))
    # shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    classifier2 = MODEL2.get_model(num_part).cuda()
    criterion = MODEL2.get_loss().cuda()
    classifier2.apply(inplace_relu)
    classifier2.load_model_from_ckpt_withrename(ckpts_masksurf)

    start_epoch = 0

    # if args.ckpts is not None:
    #     # pre_trained = torch.load(args.ckpts)['model_state_dict']
    #     # classifier.load_state_dict(pre_trained)


## we use adamw and cosine scheduler
    def add_weight_decay(model, weight_decay=1e-5, skip_list=(), optimizer_part='all'):
        decay = []
        no_decay = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if optimizer_part == 'only_new':
                if ('cls' in name):
                    if len(param.shape) == 1 or name.endswith(".bias") or 'token' in name or name in skip_list:
                        # print(name)
                        no_decay.append(param)
                    else:
                        decay.append(param)
                    print(name)
            else:
                if len(param.shape) == 1 or name.endswith(".bias") or 'token' in name or name in skip_list:
                    # print(name)
                    no_decay.append(param)
                else:
                    decay.append(param)
            # if len(param.shape) == 1 or name.endswith(".bias") or 'token' in name or name in skip_list:
            #             # print(name)
            #     no_decay.append(param)
            # else:
            #     decay.append(param)
        return [
                    {'params': no_decay, 'weight_decay': 0.},
                    {'params': decay, 'weight_decay': weight_decay}]


    param_groups = add_weight_decay(classifier, weight_decay=0.05, optimizer_part=args.optimizer_part)
    optimizer = optim.AdamW(param_groups, lr= args.learning_rate, weight_decay=0.05 )

    scheduler = CosineLRScheduler(optimizer,
                                  t_initial=args.epoch,
                                  t_mul=1,
                                  lr_min=1e-6,
                                  decay_rate=0.1,
                                  warmup_lr_init=1e-6,
                                  warmup_t=args.warmup_epoch,
                                  cycle_limit=1,
                                  t_in_epochs=True)


    classifier.zero_grad()
    for epoch in range(0, 1):
        with torch.no_grad():
            test_metrics = {}
            total_correct = 0
            total_seen = 0
            total_seen_class = [0 for _ in range(num_part)]
            total_correct_class = [0 for _ in range(num_part)]
            shape_ious = {cat: [] for cat in seg_classes.keys()}
            seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}

            for cat in seg_classes.keys():
                for label in seg_classes[cat]:
                    seg_label_to_cat[label] = cat

            classifier = classifier.eval()
            classifier2 = classifier2.eval()

            data_path = f'./vis/'
            data_path_gt = f'./vis/'
            if not os.path.exists(data_path):
                os.makedirs(data_path)
            if not os.path.exists(data_path_gt):
                os.makedirs(data_path_gt)
            selected_batch_id = [100, 300, 500, 800,  1000, 1300, 1500, 1800, 2000, 2500, 2800]
            for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader)):
                if batch_id in selected_batch_id: ## randomly select some instance for visualization.
                    cur_batch_size, NUM_POINT, _ = points.size()
                    points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
                    points = points.transpose(2, 1)
                    ### mae prediction
                    seg_pred = classifier(points, to_categorical(label, num_classes))
                    cur_pred_val = seg_pred.cpu().data.numpy()
                    cur_pred_val_logits = cur_pred_val
                    ### masksurf prediction
                    seg_pred_masksurf = classifier2(points, to_categorical(label, num_classes))
                    cur_pred_val_masksurf = seg_pred_masksurf.cpu().data.numpy()
                    cur_pred_val_logits_masksurf = cur_pred_val_masksurf

                    cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
                    target = target.cpu().data.numpy()
                    for i in range(cur_batch_size):
                        cat = seg_label_to_cat[target[i, 0]]
                        logits = cur_pred_val_logits[i, :, :]
                        logits_masksurf = cur_pred_val_logits_masksurf[i, :, :]

                        cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]
                        label_in_cate = np.argmax(logits[:, seg_classes[cat]], 1) ## 0,1,2,3
                        label_in_cate_masksurf = np.argmax(logits_masksurf[:, seg_classes[cat]], 1)  ## 0,1,2,3
                        label2color = torch.from_numpy(cmap[label_in_cate])
                        label2color_masksurf = torch.from_numpy(cmap[label_in_cate_masksurf])
                        points = points.cpu()
                        point_color = torch.cat([points[0].transpose(0,1), label2color], dim=1)
                        point_color_masksurf = torch.cat([points[0].transpose(0, 1), label2color_masksurf], dim=1)

                        gt_label_in_cate = target - seg_classes[cat][0]
                        label2color_gt = torch.from_numpy(cmap[gt_label_in_cate])
                        point_color_gt = torch.cat([points[0].transpose(0, 1), label2color_gt[0]], dim=1)

                        fout = open(data_path + cat + str(batch_id) + 'mae.obj', 'w')
                        fout_masksurf = open(data_path + cat + str(batch_id) + 'masksuf.obj', 'w')
                        fout_gt = open(data_path_gt + cat + str(batch_id) + 'gt.obj', 'w')
                        for i in range(point_color.size(0)):
                            fout.write('v %f %f %f %d %d %d\n' % (
                                point_color[i, 0], point_color[i, 1], point_color[i, 2], point_color[i, 3], point_color[i, 4],
                                point_color[i, 5]))
                            fout_masksurf.write('v %f %f %f %d %d %d\n' % (
                                point_color_masksurf[i, 0], point_color_masksurf[i, 1], point_color_masksurf[i, 2], point_color_masksurf[i, 3],
                                point_color_masksurf[i, 4],
                                point_color_masksurf[i, 5]))
                            fout_gt.write('v %f %f %f %d %d %d\n' % (
                                point_color_gt[i, 0], point_color_gt[i, 1], point_color_gt[i, 2], point_color_gt[i, 3], point_color_gt[i, 4],
                                point_color_gt[i, 5]))
                        fout.close()
                        fout_masksurf.close()
                        fout_gt.close()
                        # print((cur_pred_val == target).sum() / 2048)
                        # ipdb.set_trace()






if __name__ == '__main__':
    args = parse_args()
    main(args)