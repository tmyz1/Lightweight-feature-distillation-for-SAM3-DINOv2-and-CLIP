### 第一次，第二次提交： 测试
### 第三次提交： 新增轻量化backbone -> repvit ,以及repvit对应的蒸馏的配置文件
### 2026/8/18 
#### 1.新增vaild-mask用于解决不同尺寸上教师模型和学生模型特征层面上的对齐问题，并将DINO-V2和CLIP设置成可选项
#### 2.新增文件KD/data/get_train_and_val 用于从下载的SA-1B数据集中提取部分图像并只生成一个对应的.json文件
### 2026/8/19 
#### 1.修改limit_dataset_to_images函数中的随机种子，使得两次从大数据集中提取的图像不同
#### 2. 修改vit-small 分辨率为1008，同时保存学生模型和蒸馏模型
#### 3. 针对特征层面的损失计算，由mse替代smooth_l1
### 2026/8/20
#### 新增mask蒸馏文件，用于第二次的蒸馏，针对点提示和框提示将教师模型得到mask对学生模型的mask进行损失计算进行蒸馏
### 2026/8/25
#### 修改dataloadere文件，新增SA-1B文件，用于适配SA-1B数据集中一幅图像对应一个json文件的问题
### 2026/8/31
#### 将单教师模型蒸馏与多教师模型蒸馏分离开来

### 配置文件参数说明
```  ```
``` dataset.train_num_image ``` ``` val_num_images ``` 当训练数据集过大的时候，只采用train_num_image个训练数据集中的图片进行训练，val_num_images同理

``` dataset.max_objects_per_query ``` 一条 SAM3 的 find query 最多允许对应多少个目标实例

``` dataset.eval_category_chunk_size ``` 验证时每条 datapoint 同时包含多少个类别 query,以coco为例，80个分类，每次传给文字编码器的是80/eval_category_chunk_size个类别

``` train.return_interm_layers ``` 是否在蒸馏过程中返回中间层特征并参与损失函数的计算

```vit_small_intermediate_layers , sam3_intermediate_layers , clip_intermediate_layers , dino_v2_intermediate_layers ``` 各模型backbone之后输出的中间层特征

``` DINO_V2.is_use , CLIP.is_use ``` 是否采用DINO_V2进行特征层面的蒸馏，CLIP同理

``` train.log_every_batches , eval.log_every_batches ``` 每log_every_batches次后打印一次log

