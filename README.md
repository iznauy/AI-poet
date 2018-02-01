# AI-poet
写诗机器人

给定开头和诗歌背景创作诗句

基于pytorch

## 环境要求

pytorch 0.2+

numpy

python2.7

visdom

## 训练

训练前应当打开visdom：

`python -m visdom.server`

执行：

`python2 train.py`

命令行参数：

| 参数名（均为可选参数）     | 默认值           | 含义     |
| --------------- | ------------- | ------ |
| batch-size      | 64            |        |
| epochs          | 20            |        |
| lr              | 0.01          |        |
| data-path       | 'tang.npz'    | 数据源路径  |
| model-path      | 'checkpoint/' | 存储模型位置 |
| embedding-dim   | 128           |        |
| hidden-dim      | 256           |        |
| print-per-batch | 20            |        |

## 生成

执行：

`python2 gen.py --start-words='xxxxx'  --background='xxxx，xxxx。'` 

| 参数名（均为可选参数）   | 默认值                       | 含义                           |
| ------------- | ------------------------- | ---------------------------- |
| start-words   | '忽如一夜春风来'                 | 诗句开头                         |
| background    | '北风卷地白草折，胡天八月即飞雪。'        | 诗歌背景                         |
| max-gen-len   | 200                       | 诗歌最长生成长度                     |
| data-path     | 'tang.npz'                | 数据源路径                        |
| model-path    | 'checkpoint/tang_199.pth' | 训练好的模型位置                     |
| embedding-dim | 128                       | 选择的模型的embedding dimension    |
| hidden-dim    | 256                       | 选择的模型的LSTM层的hidden dimension |

## 备注

不支持cuda，需要使用GPU加速请自行改写代码

参考 && 数据源：

1. https://github.com/chenyuntc/pytorch-book
