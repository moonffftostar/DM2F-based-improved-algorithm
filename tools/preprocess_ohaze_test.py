import os
from PIL import Image

from config import OHAZE_ROOT
from math import ceil
from tqdm import tqdm

if __name__ == '__main__':
    ohaze_root = OHAZE_ROOT
    crop_size = 512

    ori_root = os.path.join(ohaze_root, '# O-HAZY NTIRE 2018')
    ori_haze_root = os.path.join(ori_root, 'hazy')
    ori_gt_root = os.path.join(ori_root, 'GT')

    #path for testing data
    test_root = os.path.join(ohaze_root, 'test')
    test_haze_path = os.path.join(test_root, 'hazy')
    test_gt_path = os.path.join(test_root, 'gt')

    os.makedirs(test_root, exist_ok=True)
    os.makedirs(test_haze_path, exist_ok=True)
    os.makedirs(test_gt_path, exist_ok=True)

    # last 10 images for testing
    test_list = [img_name for img_name in os.listdir(ori_haze_root)
                  if int(img_name.split('_')[0]) > 35]
    
    for idx, img_name in enumerate(tqdm(test_list)):
        gt_name = img_name.replace('hazy','GT')
        img = Image.open(os.path.join(ori_haze_root,img_name))
        gt = Image.open(os.path.join(ori_gt_root,gt_name))

        #img = img.resize((512,512),resample=0)
        #gt = gt.resize((512,512),resample=0)

        img.save(os.path.join(test_haze_path,img_name))
        gt.save(os.path.join(test_gt_path,gt_name))