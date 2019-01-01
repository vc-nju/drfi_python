<!-- 有几点需要注意的，
1.readme需要末尾四空格在换行表示换行，或者直接换行两次表示换行
2. 1.2.3.是可以有格式的
3. ![](图片链接，本地相对路径或者http)
4. Markdown Preview Enhanced. vscode可以预览markdown
5. ```python or ```c可以用制定语法高亮 -->
![](https://test-1253607195.cos.ap-shanghai.myqcloud.com/2019-1-1/logo.png)
# DRFI-python

Python realization for Saliency Object Detection: [A Discriminative Regional Feature Integration Approach](http://arxiv.org/pdf/1410.5926v1).

# Preview
![Origin_photo](https://test-1253607195.cos.ap-shanghai.myqcloud.com/2019-1-1/5.jpg)
![Saliency_map](https://test-1253607195.cos.ap-shanghai.myqcloud.com/2019-1-1/5.png)

# Feature
drfi_python is a python version for the paper mentioned above.

Some reasons you might be interested in our realization:

1. Comparing to deep learning, it's a good traditional way to realize saliency object detection.
2. The model is related to graph theory, multi-level segmentation and random forest.
3. Comparing to [CPP Version](https://github.com/playerkk/drfi_cpp) and [MATLAB Version](https://github.com/playerkk/drfi_matlab), our realization has more extensibilities because of huge python libraries.

We have trained and tested on MSRA-B, and it's auc is 0.923.

# Requirements

- python 3.x
- opencv 3.4
- scikit-image 0.14
- scikit-learn 0.20

# Installation

```bash
git clone https://github.com/vc-nju/drfi_python.git && cd drfi_python
mkdir data && mkdir data/csv && mkdir data/model && mkdir data/result
```
The pre_train models can be downloaded from [Google Drive]() and [BaiduYun](https://pan.baidu.com/s/13PJFYRRyV7uiSAosIsweTQ)(passcode: 65mp). Please copy them to data/model/

# Test Zoo

Let's take a look at a quick example.

0. Make sure you have downloaded the models and copy them to data/model/

Your data/model should be like this:
```
drfi_python
└───data
    └───model
        |  mlp.pkl
        |  rf_salience.pkl
        |  rf_same_region.pkl
```

1. Edit ./test.py module in your project:

```python
    # img_path and id can be replaced by yourself.
    img_id = 1036
    img_path = "data/MSRA-B/{}.jpg".format(img_id)
```

2. Running test using python3:
```bash
python3 test.py
```

3. Origin photo and its Saliency map are below:

![](https://test-1253607195.cos.ap-shanghai.myqcloud.com/2019-1-1/result.png)

# Training

1. Edit ./train.py in your project:

```python
    # its is your traning set's img_ids
    its = [i for i in range(1, TRAIN_IMGS + 1) if i % 5 != 0] 
    ...
    # change "data/MSRA-B/{}.jpg" to your path/to/origin_pic
    img_paths = ["data/MSRA-B/{}.jpg".format(i) for i in its] 
    # change "data/MSRA-B/{}.png" to your path/to/ground_truth_pic
    seg_paths = ["data/MSRA-B/{}.png".format(i) for i in its]
```
2. Running train using python3:

```bash
python3 train.py
```

# Validation

1. Edit ./val.py in your project:

```python
    # its is your validation set's img_ids
    its = [i for i in range(1, TRAIN_IMGS + 1) if i % 5 != 0] 
    ...
    # change "data/MSRA-B/{}.jpg" to your path/to/origin_pic
    img_paths = ["data/MSRA-B/{}.jpg".format(i) for i in its] 
    # change "data/MSRA-B/{}.png" to your path/to/ground_truth_pic
    seg_paths = ["data/MSRA-B/{}.png".format(i) for i in its]
```
2. Running validation using python3:

```bash
python3 val.py
```