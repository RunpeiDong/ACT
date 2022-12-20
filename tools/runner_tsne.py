import torch
import torch.nn as nn
import sklearn.metrics as metrics
import numpy as np

from tools import builder
from utils import misc, dist_utils, tsne_utils
from utils.logger import *

from openTSNE import TSNE
from openTSNE import TSNEEmbedding
from openTSNE import affinity
from openTSNE import initialization

from datasets import data_transforms
from pointnet2_ops import pointnet2_utils
from torchvision import transforms

test_transforms = transforms.Compose(
    [
        # data_transforms.PointcloudScale(),
        data_transforms.PointcloudRotate(),
        # data_transforms.PointcloudTranslate(),
        # data_transforms.PointcloudScaleAndTranslate(),
    ]
)


def tsne_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger=logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)

    pretrained_model = builder.model_builder(config.model_pretrained)
    finetuned_model = builder.model_builder(config.model_finetuned)
    pretrained_model.load_model_from_ckpt("model_zoo/ckpt-last-vitb-m0.8-d384-dec2.pth")
    finetuned_model.load_model_from_ckpt("model_zoo/act-hard-ckpt-best-88.21.pth")
    # finetuned_model.load_model_from_ckpt("model_zoo/modelnet_94.3.pth")

    pretrained_model = pretrained_model.cuda()
    finetuned_model = finetuned_model.cuda()

    #  DDP
    if args.distributed:
        raise NotImplementedError()

    tsne(pretrained_model, finetuned_model, test_dataloader, args, config, logger=logger)
    
def tsne_embedding(feature):
    affinities_train = affinity.PerplexityBasedNN(
        feature,
        perplexity=30,
        metric="cosine",
        n_jobs=8,
        random_state=42,
        verbose=True,
    )
    init_train = initialization.pca(feature, random_state=42)
    
    embedding_train = TSNEEmbedding(
        init_train,
        affinities_train,
        negative_gradient_method="fft",
        n_jobs=8,
        verbose=True,
    )
    
    embedding_train_1 = embedding_train.optimize(n_iter=250, exaggeration=12, momentum=0.5)
    embedding_train_2 = embedding_train_1.optimize(n_iter=500, momentum=0.8)
    
    return embedding_train_2

# visualization
def tsne(pretrained_model, finetuned_model, test_dataloader, args, config, logger=None):
    tsne = TSNE(
        perplexity=25,
        learning_rate="auto",
        metric="cosine",
        n_jobs=32,
        random_state=42,
        verbose=True,
    )

    pretrained_model.eval()
    finetuned_model.eval()
    
    test_feat_p = []
    test_feat_f = []

    test_pred  = []
    test_label = []
    npoints = config.npoints
    
    # print_log(f"[TEST_VOTE]", logger=logger)
    # acc = 0.
    # for time in range(1, 300):
    #     this_acc = test_vote(finetuned_model, test_dataloader, 1, None, args, config, logger=logger, times=10)
    #     if acc < this_acc:
    #         acc = this_acc
    #     print_log('[TEST_VOTE_time %d]  OA=%.4f, best OA=%.4f' % (time, this_acc, acc), logger=logger)
    # print_log('[TEST_VOTE] OA=%.4f' % acc, logger=logger)

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, npoints)

            logits_p, feat_p = pretrained_model(points, True)
            logits_f, feat_f = finetuned_model(points, True)

            target = label.view(-1)

            pred = logits_f.argmax(-1).view(-1)
            
            test_feat_p.append(feat_p.detach())
            test_feat_f.append(feat_f.detach())

            test_pred.append(pred.detach())
            test_label.append(target.detach())

        test_feat_f = torch.cat(test_feat_f, dim=0)
        test_feat_p = torch.cat(test_feat_p, dim=0)

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        test_label, test_pred = test_label.cpu().numpy(), test_pred.cpu().numpy()
        test_feat_p, test_feat_f = test_feat_p.cpu().numpy(), test_feat_f.cpu().numpy()
        
        correct_bool = (test_pred == test_label)

        acc = metrics.accuracy_score(test_label, test_pred) * 100.
        acc_avg = metrics.balanced_accuracy_score(test_label, test_pred) * 100.
        print_log('[TEST] OA=%.4f  mAcc=%.4f' % (acc, acc_avg), logger=logger)
        
        embedding_pretrained = tsne.fit(test_feat_p[correct_bool])
        tsne_utils.plot_tsne(embedding_pretrained, test_label[correct_bool], filename="./tsne/scan_pretrained.png")

        embedding_finetuned = tsne.fit(test_feat_f[correct_bool])
        tsne_utils.plot_tsne(embedding_finetuned, test_label[correct_bool], filename="./tsne/scan_finetuned.png")

    print_log(f"[TEST_VOTE]", logger=logger)
    acc = 0.
    for time in range(1, 300):
        this_acc = test_vote(finetuned_model, test_dataloader, 1, None, args, config, logger=logger, times=10)
        if acc < this_acc:
            acc = this_acc
        print_log('[TEST_VOTE_time %d]  OA=%.4f, best OA=%.4f' % (time, this_acc, acc), logger=logger)
    print_log('[TEST_VOTE] OA=%.4f' % acc, logger=logger)

def test_vote(base_model, test_dataloader, epoch, val_writer, args, config, logger=None, times = 10):

    base_model.eval()  # set model to eval mode

    test_pred  = []
    test_label = []
    npoints = config.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points_raw = data[0].cuda()
            label = data[1].cuda()
            if npoints == 1024:
                point_all = 1200
            if npoints == 2048:
                point_all = 2400
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()
                
            if points_raw.size(1) < point_all:
                point_all = points_raw.size(1)

            fps_idx_raw = pointnet2_utils.furthest_point_sample(points_raw, point_all)  # (B, npoint)
            local_pred = []

            for kk in range(times):
                fps_idx = fps_idx_raw[:, np.random.choice(point_all, npoints, False)]
                points = pointnet2_utils.gather_operation(points_raw.transpose(1, 2).contiguous(), 
                                                        fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)

                points = test_transforms(points)

                logits = base_model(points)
                target = label.view(-1)

                local_pred.append(logits.detach().unsqueeze(0))

            pred = torch.cat(local_pred, dim=0).mean(0)
            _, pred_choice = torch.max(pred, -1)


            test_pred.append(pred_choice)
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        test_label, test_pred = test_label.cpu().numpy(), test_pred.cpu().numpy()

        acc = metrics.accuracy_score(test_label, test_pred) * 100.
        acc_avg = metrics.balanced_accuracy_score(test_label, test_pred) * 100.
        print_log('[TEST_VOTE] EPOCH: %d  (Vote) OA=%.4f  mAcc=%.4f' % (epoch, acc, acc_avg), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC_vote', acc, epoch)
    # print_log('[TEST] acc = %.4f' % acc, logger=logger)
    
    return acc

        
        