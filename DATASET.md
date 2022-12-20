## Datasets

The overall directory structure should be:
```shell
│ACT/
├──cfgs/
├──data/
│   ├──ModelNet/
│   ├──ModelNetFewshot/
│   ├──ScanObjectNN/
│   ├──ShapeNet55-34/
│   ├──shapenetcore_partanno_segmentation_benchmark_v0_normal/
│   ├──Stanford3dDataset_v1.2_Aligned_Version/
│   ├──s3dis/
├──datasets/
├──.......
```

### ModelNet40 Dataset: 

```shell
│ModelNet/
├──modelnet40_normal_resampled/
│  ├── modelnet40_shape_names.txt
│  ├── modelnet40_train.txt
│  ├── modelnet40_test.txt
│  ├── modelnet40_train_8192pts_fps.dat
│  ├── modelnet40_test_8192pts_fps.dat
```
* Download: The data can be downloaded from [Point-BERT](https://github.com/lulutang0608/Point-BERT/blob/49e2c7407d351ce8fe65764bbddd5d9c0e0a4c52/DATASET.md), or from the [official website](https://modelnet.cs.princeton.edu/#) and processed by yourself.

### ModelNet Few-shot Dataset:
```shell
│ModelNetFewshot/
├──5way10shot/
│  ├── 0.pkl
│  ├── ...
│  ├── 9.pkl
├──5way20shot/
│  ├── ...
├──10way10shot/
│  ├── ...
├──10way20shot/
│  ├── ...
```

* Download: The data can be downloaded from [Point-BERT](https://github.com/lulutang0608/Point-BERT/blob/49e2c7407d351ce8fe65764bbddd5d9c0e0a4c52/DATASET.md). We use the same data split as theirs.

### ScanObjectNN Dataset:
```shell
│ScanObjectNN/
├──main_split/
│  ├── training_objectdataset_augmentedrot_scale75.h5
│  ├── test_objectdataset_augmentedrot_scale75.h5
│  ├── training_objectdataset.h5
│  ├── test_objectdataset.h5
├──main_split_nobg/
│  ├── training_objectdataset.h5
│  ├── test_objectdataset.h5
```
* Download: The data can be downloaded from [official website](https://hkust-vgd.github.io/scanobjectnn/).

### ShapeNet55/34 Dataset:

```shell
│ShapeNet55-34/
├──shapenet_pc_masksurf_with_normal/
│  ├── 02691156-1a04e3eab45ca15dd86060f189eb133.npy
│  ├── 02691156-1a6ad7a24bb89733f412783097373bdc.npy
│  ├── .......
├──ShapeNet-55/
│  ├── train.txt
│  └── test.txt
```

* Download: The data can be downloaded from [Point-BERT](https://github.com/lulutang0608/Point-BERT/blob/49e2c7407d351ce8fe65764bbddd5d9c0e0a4c52/DATASET.md). We use the same data split as theirs.

### ShapeNetPart Dataset:

```shell
|shapenetcore_partanno_segmentation_benchmark_v0_normal/
├──02691156/
│  ├── 1a04e3eab45ca15dd86060f189eb133.txt
│  ├── .......
│── .......
│──train_test_split/
│──synsetoffset2category.txt
```

* Download: The data can be downloaded from [official website](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip). 

### S3DIS Dataset:

```shell
|Stanford3dDataset_v1.2_Aligned_Version/
├──Area_1/
│  ├── conferenceRoom_1
│  ├── .......
│── .......
│stanford_indoor3d
│──Area_1_conferenceRoom_1.npy
│──Area_1_office_19.npy
```
Please prepare the dataset following [PointNet](https://github.com/yanx27/Pointnet_Pointnet2_pytorch):
download the `Stanford3dDataset_v1.2_Aligned_Version` from [here](http://buildingparser.stanford.edu/dataset.html), and get the processed `stanford_indoor3d` with:

```shell
cd data_utils
python collect_indoor3d_data.py
```