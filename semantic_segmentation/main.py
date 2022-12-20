"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
import torch
import time
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
from dataset import S3DISDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase',
           'board', 'clutter']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat


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
    parser.add_argument('--batch_size', type=int, default=32, help='batch Size during training')
    parser.add_argument('--epoch', default=60, type=int, help='epoch to run')
    parser.add_argument('--warmup_epoch', default=10, type=int, help='warmup epoch')
    parser.add_argument('--learning_rate', default=0.0002, type=float, help='initial learning rate')
    parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
    # parser.add_argument('--optimizer', type=str, default='AdamW', help='Adam or SGD')
    parser.add_argument('--log_dir', type=str, default='./exp', help='log path')
    # parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--npoint', type=int, default=2048, help='point Number')
    parser.add_argument('--test_area', type=int, default=5, help='test_area')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    # parser.add_argument('--step_size', type=int, default=20, help='decay step for lr decay')
    # parser.add_argument('--lr_decay', type=float, default=0.5, help='decay rate for lr decay')
    parser.add_argument('--ckpts', type=str, default=None, help='ckpts')
    parser.add_argument('--root', type=str, default='../data/stanford_indoor3d/', help='data root')
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
    exp_dir = exp_dir.joinpath('semantic_seg')
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
    logger = logging.getLogger("S3DIS")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    file_handler = logging.FileHandler('%s/%s_%s.log' % (log_dir, args.model, timestamp))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = args.root

    TRAIN_DATASET = S3DISDataset(split='train', data_root=root, num_point=args.npoint, test_area=args.test_area)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    weights = torch.Tensor(TRAIN_DATASET.labelweights).cuda()
    TEST_DATASET = S3DISDataset(split='test', data_root=root, num_point=args.npoint, test_area=args.test_area)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=10)
    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))


    num_classes = 13
    # num_part = 50

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(exp_dir))
    # shutil.copy('models/pointnet2_utils.py', str(exp_dir))

    classifier = MODEL.get_model(num_classes).cuda()
    criterion = MODEL.get_loss().cuda()
    classifier.apply(inplace_relu)
    print('# generator parameters:', sum(param.numel() for param in classifier.parameters()))
    start_epoch = 0

    if args.ckpts is not None:
        classifier.load_model_from_ckpt(args.ckpts)

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
        return [
                    {'params': no_decay, 'weight_decay': 0.},
                    {'params': decay, 'weight_decay': weight_decay}]


    param_groups = add_weight_decay(classifier, weight_decay=0.05, optimizer_part=args.optimizer_part)
    optimizer = optim.AdamW(param_groups, lr= args.learning_rate, weight_decay=0.05 )

    scheduler = CosineLRScheduler(optimizer,
                                  t_initial=args.epoch,
                                  cycle_mul=1.,
                                  lr_min=1e-6,
                                  cycle_decay=0.1,
                                  warmup_lr_init=1e-6,
                                  warmup_t=args.warmup_epoch,
                                  cycle_limit=1,
                                  t_in_epochs=True)

    best_acc = 0
    global_epoch = 0
    best_class_avg_iou = 0
    best_inctance_avg_iou = 0
    best_iou = 0
    time_sec_tot = 0.
    epoch_start_time = time.time()

    classifier.zero_grad()
    for epoch in range(start_epoch, args.epoch):
        mean_correct = []

        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        '''Adjust learning rate and BN momentum'''

        classifier = classifier.train()
        loss_batch = []
        num_iter = 0
        '''learning one epoch'''
        for i, (points, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            num_iter += 1
            points = points.data.numpy()
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points, target = points.float().cuda(),  target.long().cuda()
            points = points.transpose(2, 1)

            seg_pred = classifier(points)
            seg_pred = seg_pred.contiguous().view(-1, num_classes)
            target = target.view(-1, 1)[:, 0]
            pred_choice = seg_pred.data.max(1)[1]

            correct = pred_choice.eq(target.data).cpu().sum()
            mean_correct.append(correct.item() / (args.batch_size * args.npoint))
            loss = criterion(seg_pred, target, weights)
            loss.backward()
            optimizer.step()
            loss_batch.append(loss.detach().cpu())

            if num_iter == 1:

                torch.nn.utils.clip_grad_norm_(classifier.parameters(), 10, norm_type=2)
                num_iter = 0
                optimizer.step()
                classifier.zero_grad()

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)

        # recording time
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        time_sec_tot += epoch_time
        time_sec_avg = time_sec_tot / (
            epoch - start_epoch + 1)
        eta_sec = time_sec_avg * (args.epoch - epoch)
        eta_str = str(datetime.timedelta(seconds=int(eta_sec)))

        train_instance_acc = np.mean(mean_correct)
        loss1 = np.mean(loss_batch)
        log_string('Train accuracy is: %.5f' % (train_instance_acc * 100.0))
        log_string('Train loss: %.5f' % loss1)
        log_string('lr: %.6f' % optimizer.param_groups[0]['lr'])

        NUM_CLASSES = num_classes
        NUM_POINT = args.npoint
        BATCH_SIZE = args.batch_size

        '''Evaluate on chopped scenes'''
        with torch.no_grad():
            num_batches = len(testDataLoader)
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            labelweights = np.zeros(NUM_CLASSES)
            total_seen_class = [0 for _ in range(NUM_CLASSES)]
            total_correct_class = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
            classifier = classifier.eval()

            log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
            for i, (points, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                points = points.data.numpy()
                points = torch.Tensor(points)
                points, target = points.float().cuda(), target.long().cuda()
                points = points.transpose(2, 1)

                seg_pred = classifier(points)
                pred_val = seg_pred.contiguous().cpu().data.numpy()
                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

                batch_label = target.cpu().data.numpy()
                target = target.view(-1, 1)[:, 0]
                loss = criterion(seg_pred, target, weights)
                loss_sum += loss
                pred_val = np.argmax(pred_val, 2)
                correct = np.sum((pred_val == batch_label))
                total_correct += correct
                total_seen += (BATCH_SIZE * NUM_POINT)
                tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
                labelweights += tmp

                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum((batch_label == l))
                    total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                    total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))

            labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6))
            log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
            log_string('[mIoU] eval point avg class IoU: %f' % (mIoU * 100.0))
            log_string('[OA] eval point accuracy: %f' % (total_correct / float(total_seen) * 100.0))
            log_string('[mAcc] eval point avg class acc: %f' % (
                np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6)) * 100.0))

            iou_per_class_str = '------- IoU --------\n'
            for l in range(NUM_CLASSES):
                iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
                    seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])), labelweights[l - 1],
                    total_correct_class[l] / float(total_iou_deno_class[l]) * 100.0)

            log_string(iou_per_class_str)
            log_string('Eval mean loss: %f' % (loss_sum / num_batches))
            log_string('Eval accuracy: %f' % (total_correct / float(total_seen) * 100.0))

            if mIoU >= best_iou:
                best_iou = mIoU
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'class_avg_iou': mIoU,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_string('Saving model....')
                if mIoU >= 0.61:
                    savepath = str(checkpoints_dir) + f'/best_model_{mIoU}.pth'
                    torch.save(state, savepath)
            log_string('>>> Epoch %03d ETA: %s Best mIoU: %f' % (global_epoch + 1, eta_str, best_iou * 100.0))
            global_epoch += 1

if __name__ == '__main__':
    args = parse_args()
    main(args)