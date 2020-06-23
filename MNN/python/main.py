from voc_eval import voc_eval
import os

mAP = []
# 计算每个类别的AP
for i in range(1):
    class_name = 'person'  # 这里的类别名称为0,1,2,3,4,5,6,7
    rec, prec, ap = voc_eval('result/{}.txt', '/home/uc/RFSong-779-master/VOC2007/VOCdevkit/VOC2007/Annotations/{}.xml', 'test.txt', class_name, './')
    print("{} :\t {} ".format(class_name, ap))
    #mAP.append(ap)

#mAP = tuple(mAP)

#print("***************************")
# 输出总的mAP
#print("mAP :\t {}".format( float( sum(mAP)/len(mAP)) ))
