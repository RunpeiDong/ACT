import numpy as np
import os
from torch.utils.data import Dataset
import torch
from pointnet_util import farthest_point_sample, pc_normalize
import json


class ModelNetDataLoader(Dataset):
    def __init__(self, root, npoint=1024, split='train', uniform=False, normal_channel=True, cache_size=15000):
        self.root = root
        self.npoints = npoint
        self.uniform = uniform
        self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.normal_channel = normal_channel

        shape_ids = {}
        shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
        shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d'%(split,len(self.datapath)))

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if index in self.cache:
            point_set, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            cls = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints,:]

            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

            if not self.normal_channel:
                point_set = point_set[:, 0:3]

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls)

        return point_set, cls

    def __getitem__(self, index):
        return self._get_item(index)



import os
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset


class S3DISDataset(Dataset):
    def __init__(self, split='train', data_root='trainval_fullarea', num_point=4096, test_area=5, block_size=1.0, sample_rate=1.0, transform=None):
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform
        rooms = sorted(os.listdir(data_root))
        rooms = [room for room in rooms if 'Area_' in room]
        if split == 'train':
            rooms_split = [room for room in rooms if not 'Area_{}'.format(test_area) in room]
        else:
            rooms_split = [room for room in rooms if 'Area_{}'.format(test_area) in room]

        self.room_points, self.room_labels = [], []
        self.room_coord_min, self.room_coord_max = [], []
        num_point_all = []
        labelweights = np.zeros(13)

        for room_name in tqdm(rooms_split, total=len(rooms_split)):
            room_path = os.path.join(data_root, room_name)
            room_data = np.load(room_path)  # xyzrgbl, N*7
            points, labels = room_data[:, 0:6], room_data[:, 6]  # xyzrgb, N*6; l, N
            tmp, _ = np.histogram(labels, range(14))
            labelweights += tmp
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.room_points.append(points), self.room_labels.append(labels)
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
            num_point_all.append(labels.size)
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        print(self.labelweights)
        sample_prob = num_point_all / np.sum(num_point_all)
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point)
        room_idxs = []
        for index in range(len(rooms_split)):
            room_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
        self.room_idxs = np.array(room_idxs)
        print("Totally {} samples in {} set.".format(len(self.room_idxs), split))

    def __getitem__(self, idx):
        room_idx = self.room_idxs[idx]
        points = self.room_points[room_idx]   # N * 6
        labels = self.room_labels[room_idx]   # N
        N_points = points.shape[0]

        ## random select > 1024 samples.
        while (True):
            center = points[np.random.choice(N_points)][:3]
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
            if point_idxs.size > 1024:
                break

        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)

        # normalize
        selected_points = points[selected_point_idxs, :]  # num_point * 6
        current_points = np.zeros((self.num_point, 9))  # num_point * 9
        current_points[:, 6] = selected_points[:, 0] / self.room_coord_max[room_idx][0]
        current_points[:, 7] = selected_points[:, 1] / self.room_coord_max[room_idx][1]
        current_points[:, 8] = selected_points[:, 2] / self.room_coord_max[room_idx][2]
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        selected_points[:, 3:6] /= 255.0
        current_points[:, 0:6] = selected_points
        current_labels = labels[selected_point_idxs]
        if self.transform is not None:
            current_points, current_labels = self.transform(current_points, current_labels)
        return current_points[:, :3], current_labels

    def __len__(self):
        return len(self.room_idxs)

class ScannetDatasetWholeScene():
    # prepare to give prediction on each points
    def __init__(self, root, block_points=4096, split='test', test_area=5, stride=0.5, block_size=1.0, padding=0.001):
        self.block_points = block_points
        self.block_size = block_size
        self.padding = padding
        self.root = root
        self.split = split
        self.stride = stride
        self.scene_points_num = []
        assert split in ['train', 'test']
        if self.split == 'train':
            self.file_list = [d for d in os.listdir(root) if d.find('Area_%d' % test_area) is -1]
        else:
            self.file_list = [d for d in os.listdir(root) if d.find('Area_%d' % test_area) is not -1]
        self.scene_points_list = []
        self.semantic_labels_list = []
        self.room_coord_min, self.room_coord_max = [], []
        for file in self.file_list:
            data = np.load(root + file)
            points = data[:, :3]
            self.scene_points_list.append(data[:, :6])
            self.semantic_labels_list.append(data[:, 6])
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
        assert len(self.scene_points_list) == len(self.semantic_labels_list)

        labelweights = np.zeros(13)
        for seg in self.semantic_labels_list:
            tmp, _ = np.histogram(seg, range(14))
            self.scene_points_num.append(seg.shape[0])
            labelweights += tmp
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)

    def __getitem__(self, index):
        point_set_ini = self.scene_points_list[index]
        points = point_set_ini[:,:6]
        labels = self.semantic_labels_list[index]
        coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
        grid_x = int(np.ceil(float(coord_max[0] - coord_min[0] - self.block_size) / self.stride) + 1)
        grid_y = int(np.ceil(float(coord_max[1] - coord_min[1] - self.block_size) / self.stride) + 1)
        data_room, label_room, sample_weight, index_room = np.array([]), np.array([]), np.array([]),  np.array([])
        for index_y in range(0, grid_y):
            for index_x in range(0, grid_x):
                s_x = coord_min[0] + index_x * self.stride
                e_x = min(s_x + self.block_size, coord_max[0])
                s_x = e_x - self.block_size
                s_y = coord_min[1] + index_y * self.stride
                e_y = min(s_y + self.block_size, coord_max[1])
                s_y = e_y - self.block_size
                point_idxs = np.where(
                    (points[:, 0] >= s_x - self.padding) & (points[:, 0] <= e_x + self.padding) & (points[:, 1] >= s_y - self.padding) & (
                                points[:, 1] <= e_y + self.padding))[0]
                if point_idxs.size == 0:
                    continue
                num_batch = int(np.ceil(point_idxs.size / self.block_points))
                point_size = int(num_batch * self.block_points)
                replace = False if (point_size - point_idxs.size <= point_idxs.size) else True
                point_idxs_repeat = np.random.choice(point_idxs, point_size - point_idxs.size, replace=replace)
                point_idxs = np.concatenate((point_idxs, point_idxs_repeat))
                np.random.shuffle(point_idxs)
                data_batch = points[point_idxs, :]
                normlized_xyz = np.zeros((point_size, 3))
                normlized_xyz[:, 0] = data_batch[:, 0] / coord_max[0]
                normlized_xyz[:, 1] = data_batch[:, 1] / coord_max[1]
                normlized_xyz[:, 2] = data_batch[:, 2] / coord_max[2]
                data_batch[:, 0] = data_batch[:, 0] - (s_x + self.block_size / 2.0)
                data_batch[:, 1] = data_batch[:, 1] - (s_y + self.block_size / 2.0)
                data_batch[:, 3:6] /= 255.0
                data_batch = np.concatenate((data_batch, normlized_xyz), axis=1)
                label_batch = labels[point_idxs].astype(int)
                batch_weight = self.labelweights[label_batch]

                data_room = np.vstack([data_room, data_batch]) if data_room.size else data_batch
                label_room = np.hstack([label_room, label_batch]) if label_room.size else label_batch
                sample_weight = np.hstack([sample_weight, batch_weight]) if label_room.size else batch_weight
                index_room = np.hstack([index_room, point_idxs]) if index_room.size else point_idxs
        data_room = data_room.reshape((-1, self.block_points, data_room.shape[1]))
        label_room = label_room.reshape((-1, self.block_points))
        sample_weight = sample_weight.reshape((-1, self.block_points))
        index_room = index_room.reshape((-1, self.block_points))
        return data_room, label_room, sample_weight, index_room

    def __len__(self):
        return len(self.scene_points_list)

# if __name__ == '__main__':
    # data_root = '/data/yxu/PointNonLocal/data/stanford_indoor3d/'
    # num_point, test_area, block_size, sample_rate = 4096, 5, 1.0, 0.01
    #
    # point_data = S3DISDataset(split='train', data_root=data_root, num_point=num_point, test_area=test_area, block_size=block_size, sample_rate=sample_rate, transform=None)
    # print('point data size:', point_data.__len__())
    # print('point data 0 shape:', point_data.__getitem__(0)[0].shape)
    # print('point label 0 shape:', point_data.__getitem__(0)[1].shape)
    # import torch, time, random
    # manual_seed = 123
    # random.seed(manual_seed)
    # np.random.seed(manual_seed)
    # torch.manual_seed(manual_seed)
    # torch.cuda.manual_seed_all(manual_seed)
    # def worker_init_fn(worker_id):
    #     random.seed(manual_seed + worker_id)
    # train_loader = torch.utils.data.DataLoader(point_data, batch_size=16, shuffle=True, num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    # for idx in range(4):
    #     end = time.time()
    #     for i, (input, target) in enumerate(train_loader):
    #         print('time: {}/{}--{}'.format(i+1, len(train_loader), time.time() - end))
    #         end = time.time()

# class PartNormalDataset(Dataset):
#     def __init__(self, root='./data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', class_choice=None, normal_channel=False):
#         self.npoints = npoints
#         self.root = root
#         self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
#         self.cat = {}
#         self.normal_channel = normal_channel
#
#
#         with open(self.catfile, 'r') as f:
#             for line in f:
#                 ls = line.strip().split()
#                 self.cat[ls[0]] = ls[1]
#         self.cat = {k: v for k, v in self.cat.items()}
#         self.classes_original = dict(zip(self.cat, range(len(self.cat))))
#
#         if not class_choice is  None:
#             self.cat = {k:v for k,v in self.cat.items() if k in class_choice}
#         # print(self.cat)
#
#         self.meta = {}
#         with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
#             train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
#         with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
#             val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
#         with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
#             test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
#         for item in self.cat:
#             # print('category', item)
#             self.meta[item] = []
#             dir_point = os.path.join(self.root, self.cat[item])
#             fns = sorted(os.listdir(dir_point))
#             # print(fns[0][0:-4])
#             if split == 'trainval':
#                 fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
#             elif split == 'train':
#                 fns = [fn for fn in fns if fn[0:-4] in train_ids]
#             elif split == 'val':
#                 fns = [fn for fn in fns if fn[0:-4] in val_ids]
#             elif split == 'test':
#                 fns = [fn for fn in fns if fn[0:-4] in test_ids]
#             else:
#                 print('Unknown split: %s. Exiting..' % (split))
#                 exit(-1)
#
#             # print(os.path.basename(fns))
#             for fn in fns:
#                 token = (os.path.splitext(os.path.basename(fn))[0])
#                 self.meta[item].append(os.path.join(dir_point, token + '.txt'))
#
#         self.datapath = []
#         for item in self.cat:
#             for fn in self.meta[item]:
#                 self.datapath.append((item, fn))
#
#         self.classes = {}
#         for i in self.cat.keys():
#             self.classes[i] = self.classes_original[i]
#
#         # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
#         self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
#                             'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
#                             'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
#                             'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
#                             'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
#
#         # for cat in sorted(self.seg_classes.keys()):
#         #     print(cat, self.seg_classes[cat])
#
#         self.cache = {}  # from index to (point_set, cls, seg) tuple
#         self.cache_size = 20000
#
#
#     def __getitem__(self, index):
#         if index in self.cache:
#             point_set, cls, seg = self.cache[index]
#         else:
#             fn = self.datapath[index]
#             cat = self.datapath[index][0]
#             cls = self.classes[cat]
#             cls = np.array([cls]).astype(np.int32)
#             data = np.loadtxt(fn[1]).astype(np.float32)
#             if not self.normal_channel:
#                 point_set = data[:, 0:3]
#             else:
#                 point_set = data[:, 0:6]
#             seg = data[:, -1].astype(np.int32)
#             if len(self.cache) < self.cache_size:
#                 self.cache[index] = (point_set, cls, seg)
#         point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
#
#         choice = np.random.choice(len(seg), self.npoints, replace=True)
#         # resample
#         point_set = point_set[choice, :]
#         seg = seg[choice]
#
#         return point_set, cls, seg
#
#     def __len__(self):
#         return len(self.datapath)


# import pickle
# import os
# import sys
# import numpy as np
# import pc_util
# import scene_util
#
# class ScannetDataset(Dataset):
#     def __init__(self, root, npoints=8192, split='train'):
#         self.npoints = npoints
#         self.root = root
#         self.split = split
#         self.data_filename = os.path.join(self.root, 'scannet_%s.pickle'%(split))
#         with open(self.data_filename,'rb') as fp:
#             self.scene_points_list = pickle.load(fp)
#             self.semantic_labels_list = pickle.load(fp)
#         if split=='train':
#             labelweights = np.zeros(21)
#             for seg in self.semantic_labels_list:
#                 tmp,_ = np.histogram(seg,range(22))
#                 labelweights += tmp
#             labelweights = labelweights.astype(np.float32)
#             labelweights = labelweights/np.sum(labelweights)
#             self.labelweights = 1/np.log(1.2+labelweights)
#         elif split=='test':
#             self.labelweights = np.ones(21)
#     def __getitem__(self, index):
#         point_set = self.scene_points_list[index]
#         semantic_seg = self.semantic_labels_list[index].astype(np.int32)
#         coordmax = np.max(point_set,axis=0)
#         coordmin = np.min(point_set,axis=0)
#         smpmin = np.maximum(coordmax-[1.5,1.5,3.0], coordmin)
#         smpmin[2] = coordmin[2]
#         smpsz = np.minimum(coordmax-smpmin,[1.5,1.5,3.0])
#         smpsz[2] = coordmax[2]-coordmin[2]
#         isvalid = False
#         for i in range(10):
#             curcenter = point_set[np.random.choice(len(semantic_seg),1)[0],:]
#             curmin = curcenter-[0.75,0.75,1.5]
#             curmax = curcenter+[0.75,0.75,1.5]
#             curmin[2] = coordmin[2]
#             curmax[2] = coordmax[2]
#             curchoice = np.sum((point_set>=(curmin-0.2))*(point_set<=(curmax+0.2)),axis=1)==3
#             cur_point_set = point_set[curchoice,:]
#             cur_semantic_seg = semantic_seg[curchoice]
#             if len(cur_semantic_seg)==0:
#                 continue
#             mask = np.sum((cur_point_set>=(curmin-0.01))*(cur_point_set<=(curmax+0.01)),axis=1)==3
#             vidx = np.ceil((cur_point_set[mask,:]-curmin)/(curmax-curmin)*[31.0,31.0,62.0])
#             vidx = np.unique(vidx[:,0]*31.0*62.0+vidx[:,1]*62.0+vidx[:,2])
#             isvalid = np.sum(cur_semantic_seg>0)/len(cur_semantic_seg)>=0.7 and len(vidx)/31.0/31.0/62.0>=0.02
#             if isvalid:
#                 break
#         choice = np.random.choice(len(cur_semantic_seg), self.npoints, replace=True)
#         point_set = cur_point_set[choice,:]
#         semantic_seg = cur_semantic_seg[choice]
#         mask = mask[choice]
#         sample_weight = self.labelweights[semantic_seg]
#         sample_weight *= mask
#         return point_set, semantic_seg, sample_weight
#     def __len__(self):
#         return len(self.scene_points_list)
#
# class ScannetDatasetWholeScene():
#     def __init__(self, root, npoints=8192, split='train'):
#         self.npoints = npoints
#         self.root = root
#         self.split = split
#         self.data_filename = os.path.join(self.root, 'scannet_%s.pickle'%(split))
#         with open(self.data_filename,'rb') as fp:
#             self.scene_points_list = pickle.load(fp)
#             self.semantic_labels_list = pickle.load(fp)
#         if split=='train':
#             labelweights = np.zeros(21)
#             for seg in self.semantic_labels_list:
#                 tmp,_ = np.histogram(seg,range(22))
#                 labelweights += tmp
#             labelweights = labelweights.astype(np.float32)
#             labelweights = labelweights/np.sum(labelweights)
#             self.labelweights = 1/np.log(1.2+labelweights)
#         elif split=='test':
#             self.labelweights = np.ones(21)
#     def __getitem__(self, index):
#         point_set_ini = self.scene_points_list[index]
#         semantic_seg_ini = self.semantic_labels_list[index].astype(np.int32)
#         coordmax = np.max(point_set_ini,axis=0)
#         coordmin = np.min(point_set_ini,axis=0)
#         nsubvolume_x = np.ceil((coordmax[0]-coordmin[0])/1.5).astype(np.int32)
#         nsubvolume_y = np.ceil((coordmax[1]-coordmin[1])/1.5).astype(np.int32)
#         point_sets = list()
#         semantic_segs = list()
#         sample_weights = list()
#         isvalid = False
#         for i in range(nsubvolume_x):
#             for j in range(nsubvolume_y):
#                 curmin = coordmin+[i*1.5,j*1.5,0]
#                 curmax = coordmin+[(i+1)*1.5,(j+1)*1.5,coordmax[2]-coordmin[2]]
#                 curchoice = np.sum((point_set_ini>=(curmin-0.2))*(point_set_ini<=(curmax+0.2)),axis=1)==3
#                 cur_point_set = point_set_ini[curchoice,:]
#                 cur_semantic_seg = semantic_seg_ini[curchoice]
#                 if len(cur_semantic_seg)==0:
#                     continue
#                 mask = np.sum((cur_point_set>=(curmin-0.001))*(cur_point_set<=(curmax+0.001)),axis=1)==3
#                 choice = np.random.choice(len(cur_semantic_seg), self.npoints, replace=True)
#                 point_set = cur_point_set[choice,:] # Nx3
#                 semantic_seg = cur_semantic_seg[choice] # N
#                 mask = mask[choice]
#                 if sum(mask)/float(len(mask))<0.01:
#                     continue
#                 sample_weight = self.labelweights[semantic_seg]
#                 sample_weight *= mask # N
#                 point_sets.append(np.expand_dims(point_set,0)) # 1xNx3
#                 semantic_segs.append(np.expand_dims(semantic_seg,0)) # 1xN
#                 sample_weights.append(np.expand_dims(sample_weight,0)) # 1xN
#         point_sets = np.concatenate(tuple(point_sets),axis=0)
#         semantic_segs = np.concatenate(tuple(semantic_segs),axis=0)
#         sample_weights = np.concatenate(tuple(sample_weights),axis=0)
#         return point_sets, semantic_segs, sample_weights
#     def __len__(self):
#         return len(self.scene_points_list)
#
# class ScannetDatasetVirtualScan():
#     def __init__(self, root, npoints=8192, split='train'):
#         self.npoints = npoints
#         self.root = root
#         self.split = split
#         self.data_filename = os.path.join(self.root, 'scannet_%s.pickle'%(split))
#         with open(self.data_filename,'rb') as fp:
#             self.scene_points_list = pickle.load(fp)
#             self.semantic_labels_list = pickle.load(fp)
#         if split=='train':
#             labelweights = np.zeros(21)
#             for seg in self.semantic_labels_list:
#                 tmp,_ = np.histogram(seg,range(22))
#                 labelweights += tmp
#             labelweights = labelweights.astype(np.float32)
#             labelweights = labelweights/np.sum(labelweights)
#             self.labelweights = 1/np.log(1.2+labelweights)
#         elif split=='test':
#             self.labelweights = np.ones(21)
#     def __getitem__(self, index):
#         point_set_ini = self.scene_points_list[index]
#         semantic_seg_ini = self.semantic_labels_list[index].astype(np.int32)
#         sample_weight_ini = self.labelweights[semantic_seg_ini]
#         point_sets = list()
#         semantic_segs = list()
#         sample_weights = list()
#         for i in xrange(8):
#             smpidx = scene_util.virtual_scan(point_set_ini,mode=i)
#             if len(smpidx)<300:
#                 continue
#             point_set = point_set_ini[smpidx,:]
#             semantic_seg = semantic_seg_ini[smpidx]
#             sample_weight = sample_weight_ini[smpidx]
#             choice = np.random.choice(len(semantic_seg), self.npoints, replace=True)
#             point_set = point_set[choice,:] # Nx3
#             semantic_seg = semantic_seg[choice] # N
#             sample_weight = sample_weight[choice] # N
#             point_sets.append(np.expand_dims(point_set,0)) # 1xNx3
#             semantic_segs.append(np.expand_dims(semantic_seg,0)) # 1xN
#             sample_weights.append(np.expand_dims(sample_weight,0)) # 1xN
#         point_sets = np.concatenate(tuple(point_sets),axis=0)
#         semantic_segs = np.concatenate(tuple(semantic_segs),axis=0)
#         sample_weights = np.concatenate(tuple(sample_weights),axis=0)
#         return point_sets, semantic_segs, sample_weights
#     def __len__(self):
#         return len(self.scene_points_list)

# if __name__=='__main__':
#     d = ScannetDatasetWholeScene(root = './data', split='test', npoints=8192)
#     labelweights_vox = np.zeros(21)
#     for ii in xrange(len(d)):
#         print ii
#         ps,seg,smpw = d[ii]
#         for b in xrange(ps.shape[0]):
#             _, uvlabel, _ = pc_util.point_cloud_label_to_surface_voxel_label_fast(ps[b,smpw[b,:]>0,:], seg[b,smpw[b,:]>0], res=0.02)
#         tmp,_ = np.histogram(uvlabel,range(22))
#         labelweights_vox += tmp
#     print labelweights_vox[1:].astype(np.float32)/np.sum(labelweights_vox[1:].astype(np.float32))
#     exit()




if __name__ == '__main__':
    data = ModelNetDataLoader('modelnet40_normal_resampled/', split='train', uniform=False, normal_channel=True)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point,label in DataLoader:
        print(point.shape)
        print(label.shape)