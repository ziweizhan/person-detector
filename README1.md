# person-detector    
## 环境搭建    
1. 首先安装pytorch环境。pytorch版本1.2及以上版本。 
```python
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch //安装最新pytorch1.5 
```
2. 安装编译MNN。    
2.1. [MNN安装过程请参考我的博客](https://blog.csdn.net/donkey_1993/article/details/106378976)    
2.2. 安装包安装完成之后会在`MNN/build`文件夹里面生成一个`libMNN.so`文件，需要将这个文件复制到`person-detector/MNN/mnn/lib`进行替换。
3. 其他安装包安装。
```python
conda install 安装包
or
pip install 安装包
```
## 测试过程
### python测试教程
1. 直接运行`detect_imgs.py`就可以检测`imgs`文件夹里面的图片。    
```python
python detect_imgs.py
```
### MNN中测试过程
#### 首先需要对person-detector/MNN进行编译    
1. 将person-detector/MNN/build删除。
2. 然后建立build文件，重新进行编译。
```person
cd person-detector/MNN
mkdir build
cd build
cmake ..
make -j8
```
#### 运行MNN中python版本  
```python
1. cd person-detector/MNN/python
2. python person-detector-pic.py --imgs_path ../imgs         # 检测图片，可以在person-detector-pic.py里面修改图片测试路径。
3. python person-detector-video.py  # 检测视频，可以在person-detector-video.py里面修改视频测试路径。
```
#### 运行MNN中C++版本
```bash
cd build
./Ultra-face-mnn ../model/version-RFB/617-1.mnn ../imgs/timg.jpg
```
```bash
# 量化之后的版本
cd build
./Ultra-face-mnn ../model/version-RFB/617-1-sq.mnn ../imgs/timg.jpg
```
## 训练过程    
### 数据集准备    
1. 训练数据集采用的是VOC格式的数据集。训练数据集路径可以在`train-version-RFB.sh`里面修改。    
2. 使用脚本可以将`coco`数据集转成`VOC`格式且只有行人检测框的数据集。    
[COCO数据集转VOC数据集只包含行人的转换教程](https://blog.csdn.net/donkey_1993/article/details/106279988)
### 运行训练
```python
bash train-version-RFB.sh
```
## 模型转换  
1. pth模型转onnx模型，在转模型之间需要将person-detector/version/ssd/ssd.py进行修改,修改成下面的样子。    
```python
        if self.is_test:
            confidences = F.softmax(confidences, dim=2)
            boxes = locations
            #boxes = box_utils.convert_locations_to_boxes(
            #    locations, self.priors, self.config.center_variance, self.config.size_variance
            #)
            #boxes = box_utils.center_form_to_corner_form(boxes)
            return confidences, boxes
```
然后再运行转换文件。
```python
python convert_to_onnx.py
```
2. 将onnx模型转成mnn模型    
2.1 进到主目录下的MNN/build文件里面（不是person-detector/MNN文件）    
2.2 运行MNNConvert来转换模型
```python
./MNNConvert -f ONNX --modelFile XXX.onnx --MNNModel XXX.mnn --bizCode biz
```
3. 模型的INT8量化
3.1 编译MNN的量化工具。
```python
cd MNN/build
cmake .. -DMNN_BUILD_QUANTOOLS=on
make -j8
```
3.2 构建一个pretreatConfig.json文件，代码如下：
```python
{
    "format":"RGB",
    "mean":[
        127.5,
        127.5,
        127.5
    ],
    "normal":[
        0.00784314,
        0.00784314,
        0.00784314
    ],
    "width":224,
    "height":224,
    "path":"path/to/images/",
    "used_image_num":500,
    "feature_quantize_method":"KL",
    "weight_quantize_method":"MAX_ABS"
}
```
3.3 在MNN/build里面运行量化程序。    
```python
./quantized.out XXX.mnn XXX-INT8.mnn pretreatConfig.json
```
