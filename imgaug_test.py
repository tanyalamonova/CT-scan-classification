import glob
import cv2
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import imageio

ia.seed(1)

seq = iaa.Sequential(
    [
        iaa.Fliplr(0.5),                                  
        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0.1, 3.0))),     
        iaa.ContrastNormalization((0.5, 1.5)),                   
        iaa.AdditiveGaussianNoise(
            loc=0, scale=(0.0, 0.1 * 255), per_channel=0),     
        iaa.Multiply((0.8, 1.2), per_channel=0),                
        iaa.Affine(
            scale={
                "x": (0.8, 1.2),
                "y": (0.8, 1.2)
            },
            translate_percent={
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1)
            },
            rotate=(-10, 10))                                       
    ],
    random_order=True)



#region mult images
image_list = [cv2.imread(file) for file in glob.glob('test/abnormal/*.png')]
for index, img in enumerate(image_list):
    images = np.array([img for _ in range(64)], dtype=np.uint8) 
    images_aug = seq.augment_images(images)
    for i in range(64):
        imageio.imwrite('test/abnormal/'+str(index)+'_'+str(i)+'new.png', images_aug[i])
#endregion
