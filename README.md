### 配置文件参数说明
```  ```
``` dataset.train_num_image ``` ``` val_num_images ``` 当训练数据集过大的时候，只采用train_num_image个训练数据集中的图片进行训练，val_num_images同理

``` dataset.max_objects_per_query ``` 一条 SAM3 的 find query 最多允许对应多少个目标实例

``` dataset.eval_category_chunk_size ``` 验证时每条 datapoint 同时包含多少个类别 query,以coco为例，80个分类，每次传给文字编码器的是80/eval_category_chunk_size个类别

``` train.return_interm_layers ``` 是否在蒸馏过程中返回中间层特征并参与损失函数的计算

```vit_small_intermediate_layers , sam3_intermediate_layers , clip_intermediate_layers , dino_v2_intermediate_layers ``` 各模型backbone之后输出的中间层特征

``` DINO_V2.is_use , CLIP.is_use ``` 是否采用DINO_V2进行特征层面的蒸馏，CLIP同理

``` train.log_every_batches , eval.log_every_batches ``` 每log_every_batches次后打印一次log

### 有关配置文件的参数说明，我放在了[这里](Vit_Small_Distill_server.yaml)

### 如果要更换backbone，可前往[model/__init__](KD/model/__init__.py)中查看对应需要的步骤

### 关于训练，如果只采用neck和feature两个部分的损失，在超参数，也就是这两个部分的权重分配上，尽可能在完成一轮训练之后的损失值比例相近，目前来看，feature：neck = 1：4的权重
### 所得到的效果最好

### 关于损失函数，经过对比，在同一个框架下面（vit）采用L2，mse的损失函数计算所得到的效果是最好的,关于不同框架，阅读文献，在L2损失的基础上增加cosine损失函数能取得更好的效果，但是
### 大部分这类文献基本都采用了大量的数据更长的时间来进行训练。


### 根据文献 VITKD，主要是针对vit 到 vit 框架的蒸馏，在vit框架特征提取的浅层采用linear层进行对齐，在深层采用深层式对齐[Generation_adapter](KD/KD_Loss.py)能够得到更好的效果
### 经过实验，我们采用了sam3的[7,15,23,31]层 和vit_small 的 [2, 5, 8, 11]进行对齐，其中前两层采用的是linear对齐，后两层是Generation对齐，并取得了更好的精度

### 关于软监督，主要采用的是sam3的logit输出与学生模型的输出来作对比，与backbone部分的蒸馏不同的是，这部分的蒸馏是明确需要文字分类部分的，如coco数据集针对每一个目标，
### 都有对应的具体的类，而sa1b数据集则没有，比较合理的猜想自己设定一定的分类，写一个脚本通过sam3模型去推理生成有明确分类的.json文件，后续用于蒸馏训练



