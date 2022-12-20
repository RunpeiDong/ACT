## Model Zoo
The models and logs can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1hZUmqRvAg64abnkaI1HxctfQPTetWpaH?usp=share_link). You can download them and place the pretrained weights here.

### ACT Pretrained Models

| Task                  | Dataset                | Config                                                       | Acc.       | Download                                                     |
| --------------------- | ---------------------- | ------------------------------------------------------------ | ---------- | ------------------------------------------------------------ |
| Autoencoding          | ShapeNet               | [act_dvae_with_pretrained_transformer.yaml](../cfgs/autoencoder/act_dvae_with_pretrained_transformer.yaml) | N/A        | [here](https://drive.google.com/file/d/1Q-QAsaJQI-3Wci6BHKjcLbhIMkT-YQKQ/view?usp=share_link) |
| Pre-training          | ShapeNet               | [pretrain_act_distill.yaml](../cfgs/pretrain/pretrain_act_distill.yaml) | N/A        | [here](https://drive.google.com/file/d/1T8bzdJfzdfQtCLu3WU9yDZTgBrLXSDcE/view?usp=share_link) |
| Classification        | ScanObjectNN PB_T50_RS | [finetune_scan_hardest.yaml](../cfgs/finetune_classification/full/finetune_scan_hardest.yaml) | 88.21%     | [here](https://drive.google.com/file/d/1HgxlISsaJrBMOvhhHt7D92wGFlWTb8uw/view?usp=share_link) |
| Classification        | ScanObjectNN OBJ_BG    | [finetune_scan_objbg.yaml](./cfgs/finetune_classification/full/finetune_scan_objbg.yaml) | 93.29%     | [here](https://drive.google.com/file/d/1JN5DzZKQLGZo8Rxxp3SdPYIwB5cilhVW/view?usp=share_link) |
| Classification        | ScanObjectNN OBJ_ONLY  | [finetune_scan_objonly.yaml](./cfgs/finetune_classification/full/finetune_scan_objonly.yaml) | 91.91%     | [here](https://drive.google.com/file/d/1dr79MA5HrRS3NHbzXQprfjoAS1Ou-U78/view?usp=share_link) |
| Classification        | ModelNet40             | [finetune_modelnet.yaml](../cfgs/finetune_classification/full/finetune_modelnet.yaml) | 93.70%     | [here](https://drive.google.com/file/d/1SaQI9npzTHf5Ty61vLOORKXjDo1c_yhR/view?usp=share_link) |
| Semantic Segmentation | S3DIS                  | [semantic_segmentation](../semantic_segmentation)            | 61.2% mIoU | [here](https://drive.google.com/file/d/11JXdCkT91A2vdjBsw8-sroYZTryek7a9/view?usp=share_link) |

| Task              | Dataset    | Config                                                       | 5w10s Acc. (%) | 5w20s Acc. (%) | 10w10s Acc. (%) | 10w20s Acc. (%) |
| ----------------- | ---------- | ------------------------------------------------------------ | -------------- | -------------- | --------------- | --------------- |
| Few-Shot Learning | ModelNet40 | [fewshot_modelnet.yaml](../cfgs/finetune_classification/few_shot/fewshot_modelnet.yaml) | 96.8 ± 2.3     | 98.0 ± 1.4     | 93.3 ± 4.0      | 95.6 ± 2.8      |
