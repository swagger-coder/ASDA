__author__ = 'licheng'

"""
This interface provides access to four datasets:
1) refclef
2) refcoco
3) refcoco+
4) refcocog
split by unc and google
The following API functions are defined:
REFER      - REFER api class
getRefIds  - get ref ids that satisfy given filter conditions.
getAnnIds  - get ann ids that satisfy given filter conditions.
getImgIds  - get image ids that satisfy given filter conditions.
getCatIds  - get category ids that satisfy given filter conditions.
loadRefs   - load refs with the specified ref ids.
loadAnns   - load anns with the specified ann ids.
loadImgs   - load images with the specified image ids.
loadCats   - load category names with the specified category ids.
getRefBox  - get ref's bounding box [x, y, w, h] given the ref_id
showRef    - show image, segmentation or box of the referred object with the ref
getMask    - get mask and area of the referred object given ref
showMask   - show mask of the referred object given ref
"""

import sys
import os.path as osp
import os
import json
# import _pickle  as pickle
import pickle
import time
import itertools
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
from pprint import pprint
import numpy as np
from pycocotools import mask
import cv2
# from skimage.measure import label, regionprops

class REFER:
    def __init__(self, data_root, dataset='refcoco', splitBy='unc'):
        # provide data_root folder which contains refclef, refcoco, refcoco+ and refcocog
        # also provide dataset name and splitBy information
        # e.g., dataset = 'refcoco', splitBy = 'unc'
        print('loading dataset %s into memory...' % dataset)
        self.ROOT_DIR = osp.abspath(osp.dirname(__file__))
        self.DATA_DIR = osp.join(data_root, dataset)
        if dataset in ['refcoco', 'refcoco+', 'refcocog']:
            self.IMAGE_DIR = osp.join(data_root, 'images/train2014')
        elif dataset == 'refclef':
            self.IMAGE_DIR = osp.join(data_root, 'images/saiapr_tc-12')
        else:
            print('No refer dataset is called [%s]' % dataset)
            sys.exit()

        # load refs from data/dataset/refs(dataset).json
        tic = time.time()
        ref_file = osp.join(self.DATA_DIR, 'refs('+splitBy+').p')
        self.data = {}
        self.data['dataset'] = dataset

        self.data['refs'] = pickle.load(open(ref_file, 'rb'),fix_imports=True)

        # load annotations from data/dataset/instances.json
        instances_file = osp.join(self.DATA_DIR, 'instances.json')
        instances = json.load(open(instances_file, 'r'))
        self.data['images'] = instances['images']
        self.data['annotations'] = instances['annotations']
        self.data['categories'] = instances['categories']

        # create index
        self.createIndex()
        print('DONE (t=%.2fs)' % (time.time()-tic))

    def createIndex(self):
        # create sets of mapping
        # 1)  Refs: 	 	{ref_id: ref}
        # 2)  Anns: 	 	{ann_id: ann}
        # 3)  Imgs:		 	{image_id: image}
        # 4)  Cats: 	 	{category_id: category_name}
        # 5)  Sents:     	{sent_id: sent}
        # 6)  imgToRefs: 	{image_id: refs}
        # 7)  imgToAnns: 	{image_id: anns}
        # 8)  refToAnn:  	{ref_id: ann}
        # 9)  annToRef:  	{ann_id: ref}
        # 10) catToRefs: 	{category_id: refs}
        # 11) sentToRef: 	{sent_id: ref}
        # 12) sentToTokens: {sent_id: tokens}
        print('creating index...')
        # fetch info from instances
        Anns, Imgs, Cats, imgToAnns = {}, {}, {}, {}
        for ann in self.data['annotations']:
            Anns[ann['id']] = ann
            imgToAnns[ann['image_id']] = imgToAnns.get(ann['image_id'], []) + [ann]
        for img in self.data['images']:
            Imgs[img['id']] = img
        for cat in self.data['categories']:
            Cats[cat['id']] = cat['name']

        # fetch info from refs
        Refs, imgToRefs, refToAnn, annToRef, catToRefs = {}, {}, {}, {}, {}
        Sents, sentToRef, sentToTokens = {}, {}, {}
        for ref in self.data['refs']:
            # ids
            ref_id = ref['ref_id']
            ann_id = ref['ann_id']
            category_id = ref['category_id']
            image_id = ref['image_id']

            # add mapping related to ref
            Refs[ref_id] = ref
            imgToRefs[image_id] = imgToRefs.get(image_id, []) + [ref]
            catToRefs[category_id] = catToRefs.get(category_id, []) + [ref]
            refToAnn[ref_id] = Anns[ann_id]
            annToRef[ann_id] = ref

            # add mapping of sent
            for sent in ref['sentences']:
                Sents[sent['sent_id']] = sent
                sentToRef[sent['sent_id']] = ref
                sentToTokens[sent['sent_id']] = sent['tokens']

        # create class members
        self.Refs = Refs
        self.Anns = Anns
        self.Imgs = Imgs
        self.Cats = Cats
        self.Sents = Sents
        self.imgToRefs = imgToRefs
        self.imgToAnns = imgToAnns
        self.refToAnn = refToAnn
        self.annToRef = annToRef
        self.catToRefs = catToRefs
        self.sentToRef = sentToRef
        self.sentToTokens = sentToTokens
        print('index created.')

    def getRefIds(self, image_ids=[], cat_ids=[], ref_ids=[], split=''):
        image_ids = image_ids if type(image_ids) == list else [image_ids]
        cat_ids = cat_ids if type(cat_ids) == list else [cat_ids]
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if len(image_ids)==len(cat_ids)==len(ref_ids)==len(split)==0:
            refs = self.data['refs']
        else:
            if not len(image_ids) == 0:
                refs = [self.imgToRefs[image_id] for image_id in image_ids]
            else:
                refs = self.data['refs']
            if not len(cat_ids) == 0:
                refs = [ref for ref in refs if ref['category_id'] in cat_ids]
            if not len(ref_ids) == 0:
                refs = [ref for ref in refs if ref['ref_id'] in ref_ids]
            if not len(split) == 0:
                if split in ['testA', 'testB', 'testC']:
                    refs = [ref for ref in refs if split[-1] in ref['split']] # we also consider testAB, testBC, ...
                elif split in ['testAB', 'testBC', 'testAC']:
                    refs = [ref for ref in refs if ref['split'] == split]  # rarely used I guess...
                elif split == 'test':
                    refs = [ref for ref in refs if 'test' in ref['split']]
                elif split == 'train' or split == 'val':
                    refs = [ref for ref in refs if ref['split'] == split]
                else:
                    print('No such split [%s]' % split)
                    sys.exit()
        ref_ids = [ref['ref_id'] for ref in refs]
        return ref_ids

    def getAnnIds(self, image_ids=[], cat_ids=[], ref_ids=[]):
        image_ids = image_ids if type(image_ids) == list else [image_ids]
        cat_ids = cat_ids if type(cat_ids) == list else [cat_ids]
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if len(image_ids) == len(cat_ids) == len(ref_ids) == 0:
            ann_ids = [ann['id'] for ann in self.data['annotations']]
        else:
            if not len(image_ids) == 0:
                lists = [self.imgToAnns[image_id] for image_id in image_ids if image_id in self.imgToAnns]  # list of [anns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.data['annotations']
            if not len(cat_ids) == 0:
                anns = [ann for ann in anns if ann['category_id'] in cat_ids]
            ann_ids = [ann['id'] for ann in anns]
            if not len(ref_ids) == 0:
                ids = set(ann_ids).intersection(set([self.Refs[ref_id]['ann_id'] for ref_id in ref_ids]))
        return ann_ids

    def getImgIds(self, ref_ids=[]):
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if not len(ref_ids) == 0:
            image_ids = list(set([self.Refs[ref_id]['image_id'] for ref_id in ref_ids]))
        else:
            image_ids = self.Imgs.keys()
        return image_ids

    def getCatIds(self):
        return self.Cats.keys()

    def loadRefs(self, ref_ids=[]):
        if type(ref_ids) == list:
            return [self.Refs[ref_id] for ref_id in ref_ids]
        elif type(ref_ids) == int:
            return [self.Refs[ref_ids]]

    def loadAnns(self, ann_ids=[]):
        if type(ann_ids) == list:
            return [self.Anns[ann_id] for ann_id in ann_ids]
        elif type(ann_ids) == int or type(ann_ids) == unicode:
            return [self.Anns[ann_ids]]

    def loadImgs(self, image_ids=[]):
        if type(image_ids) == list:
            return [self.Imgs[image_id] for image_id in image_ids]
        elif type(image_ids) == int:
            return [self.Imgs[image_ids]]

    def loadCats(self, cat_ids=[]):
        if type(cat_ids) == list:
            return [self.Cats[cat_id] for cat_id in cat_ids]
        elif type(cat_ids) == int:
            return [self.Cats[cat_ids]]

    def getRefBox(self, ref_id):
        ref = self.Refs[ref_id]
        ann = self.refToAnn[ref_id]
        return ann['bbox']  # [x, y, w, h]

    def showRef(self, ref, seg_box='seg'):
        ax = plt.gca()
        # show image
        image = self.Imgs[ref['image_id']]
        I = io.imread(osp.join(self.IMAGE_DIR, image['file_name']))
        ax.imshow(I)
        # show refer expression
        for sid, sent in enumerate(ref['sentences']):
            print('%s. %s' % (sid+1, sent['sent']))
        # show segmentations
        if seg_box == 'seg':
            ann_id = ref['ann_id']
            ann = self.Anns[ann_id]
            polygons = []
            color = []
            c = 'none'
            if type(ann['segmentation'][0]) == list:
                # polygon used for refcoco*
                for seg in ann['segmentation']:
                    poly = np.array(seg).reshape((len(seg)//2, 2))
                    polygons.append(Polygon(poly, True, alpha=0.4))
                    color.append(c)
                p = PatchCollection(polygons, facecolors=color, edgecolors=(1,1,0,0), linewidths=3, alpha=1)
                ax.add_collection(p)  # thick yellow polygon
                p = PatchCollection(polygons, facecolors=color, edgecolors=(1,0,0,0), linewidths=1, alpha=1)
                ax.add_collection(p)  # thin red polygon
            else:
                # mask used for refclef
                rle = ann['segmentation']
                m = mask.decode(rle)
                img = np.ones( (m.shape[0], m.shape[1], 3) )
                color_mask = np.array([2.0,166.0,101.0])/255
                for i in range(3):
                    img[:,:,i] = color_mask[i]
                ax.imshow(np.dstack( (img, m*0.5) ))
        # show bounding-box
        elif seg_box == 'box':
            ann_id = ref['ann_id']
            print(ann_id)
            ann = self.Anns[ann_id]
            bbox = 	self.getRefBox(ref['ref_id'])
            box_plot = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], fill=False, edgecolor='green', linewidth=3)
            ax.add_patch(box_plot)

    def getMask(self, ref):
        # return mask, area and mask-center
        ann = self.refToAnn[ref['ref_id']]
        print(ann)
        image = self.Imgs[ref['image_id']]
        if type(ann['segmentation'][0]) == list: # polygon
            rle = mask.frPyObjects(ann['segmentation'], image['height'], image['width'])
        else:
            rle = ann['segmentation']

        # for i in range(len(rle['counts'])):
        # print(rle)
        m = mask.decode(rle)
        m = np.sum(m, axis=2)  # sometimes there are multiple binary map (corresponding to multiple segs)
        m = m.astype(np.uint8) # convert to np.uint8
        # compute area
        area = sum(mask.area(rle))  # should be close to ann['area']
        return {'mask': m, 'area': area}
        # # position
        # position_x = np.mean(np.where(m==1)[1]) # [1] means columns (matlab style) -> x (c style)
        # position_y = np.mean(np.where(m==1)[0]) # [0] means rows (matlab style)    -> y (c style)
        # # mass position (if there were multiple regions, we use the largest one.)
        # label_m = label(m, connectivity=m.ndim)
        # regions = regionprops(label_m)
        # if len(regions) > 0:
        # 	largest_id = np.argmax(np.array([props.filled_area for props in regions]))
        # 	largest_props = regions[largest_id]
        # 	mass_y, mass_x = largest_props.centroid
        # else:
        # 	mass_x, mass_y = position_x, position_y
        # # if centroid is not in mask, we find the closest point to it from mask
        # if m[mass_y, mass_x] != 1:
        # 	print 'Finding closes mask point ...'
        # 	kernel = np.ones((10, 10),np.uint8)
        # 	me = cv2.erode(m, kernel, iterations = 1)
        # 	points = zip(np.where(me == 1)[0].tolist(), np.where(me == 1)[1].tolist())  # row, col style
        # 	points = np.array(points)
        # 	dist   = np.sum((points - (mass_y, mass_x))**2, axis=1)
        # 	id     = np.argsort(dist)[0]
        # 	mass_y, mass_x = points[id]
        # 	# return
        # return {'mask': m, 'area': area, 'position_x': position_x, 'position_y': position_y, 'mass_x': mass_x, 'mass_y': mass_y}
        # # show image and mask
        # I = io.imread(osp.join(self.IMAGE_DIR, image['file_name']))
        # plt.figure()
        # plt.imshow(I)
        # ax = plt.gca()
        # img = np.ones( (m.shape[0], m.shape[1], 3) )
        # color_mask = np.array([2.0,166.0,101.0])/255
        # for i in range(3):
        #     img[:,:,i] = color_mask[i]
        # ax.imshow(np.dstack( (img, m*0.5) ))
        # plt.show()

    def showMask(self, ref):
        M = self.getMask(ref)
        msk = M['mask']
        ax = plt.gca()
        ax.imshow(msk)


if __name__ == '__main__':
    refer = REFER(data_root="/home/ypf/workspace/code/BKINet/ln_data", dataset='refcoco', splitBy='unc')
    save_path = "./visualization/"
    ref_ids = refer.getRefIds()
    print(len(ref_ids))

    print(len(refer.Imgs))
    print(len(refer.imgToRefs))
    print(refer.Cats)

    ref_ids = refer.getRefIds(split='train')
    print('There are %s training referred objects.' % len(ref_ids))

    img_ids = [8936, 52563]
    # ref_ids = refer.getRefIds(image_ids=img_ids)

    # refs = refer.loadRefs(ref_ids)

    def custom_vis1(image, mask_):
        # 将mask应用到蓝色图层
        # 创建一个蓝色图层
        blue_layer = np.zeros_like(image)
        blue_layer[:, :, 0] = 255  # 对于OpenCV，蓝色通道是第一个
        blue_mask = cv2.bitwise_and(blue_layer, blue_layer, mask=mask_)

        # 将蓝色mask以一定的透明度覆盖到原图上
        alpha = 0.1  # 透明度
        cv2.addWeighted(blue_mask, alpha, image, 1 - alpha, 0, image)
    
    def custom_vis2(image, mask_):
        # 创建蓝色图层
        blue_layer = np.zeros_like(image)
        blue_layer[:, :, 0] = 255  # 对于OpenCV，蓝色通道是第一个

        # 将mask应用到蓝色图层
        blue_mask = cv2.bitwise_and(blue_layer, blue_layer, mask=mask_)

        # alpha值定义了mask图层和原图的融合程度
        alpha = 0.5  # 透明度

        # 创建一个完全透明的图层
        transparent_layer = np.zeros_like(image)

        # 我们只在mask的区域上应用蓝色图层，并调整alpha值来控制透明度
        for i in range(3):  # 只处理RGB三个通道
            transparent_layer[:, :, i] = cv2.addWeighted(blue_mask[:, :, i], alpha, image[:, :, i], 1 - alpha, 0)

        # 在mask区域外使用原图
        transparent_layer[mask_ == 0] = image[mask_ == 0]

        return transparent_layer

    def custom_vis3(image, mask_):
        """
        直接在原图上修改指定mask区域的颜色为蓝色
        不改变其他区域的亮度或色彩
        """
        image[mask_ != 0] = [255, 0, 0]  # OpenCV中的颜色顺序是BGR
    
    def custom_vis4(image, mask_, alpha=0.4):
        """
        在原图上以指定的透明度应用蓝色遮罩。
        alpha: 遮罩的透明度，范围从0（完全透明）到1（完全不透明）。
        """
        # 将原图从BGR转换为RGBA以添加Alpha通道
        image_rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        # 创建一个同样大小的全蓝色图层
        blue_mask = np.zeros_like(image_rgba)
        blue_mask[:, :, 0] = 255  # B
        blue_mask[:, :, 3] = 255  # Alpha设置为不透明

        # 应用透明度到mask区域
        blue_mask[mask_ != 0, 3] = int(alpha * 255)

        # 将蓝色遮罩叠加到原图
        image_rgba = cv2.addWeighted(image_rgba, 1, blue_mask, alpha, 0)
        return image_rgba




    for i, img_id in enumerate(img_ids):
        ref = refer.imgToRefs[img_id][0]
        print(ref)
        mask_ = refer.getMask(ref)['mask']
        # sentence = ref['sentences'][0]['sent']

        img = refer.Imgs[img_id]
        # I = io.imread(osp.join(refer.IMAGE_DIR, img['file_name']))
        # 假设`image_path`是原始图像的路径，`mask`是一个与原图像相同大小的二值数组
        image_path = osp.join(refer.IMAGE_DIR, img['file_name'])
        image = cv2.imread(image_path)
        # mask = np.zeros(image.shape[:2], dtype=np.uint8)  # 这里你需要有一个实际的mask

        # custom_vis1(image, mask_)

        image = custom_vis2(image, mask_)

        # custom_vis3(image, mask_)  

        # image = custom_vis4(image=image, mask_=mask_, alpha=0.4) 
        

        

        # 保存结果图像到指定路径
        image_dir = osp.join(save_path, str(img_id))
        osp.exists(image_dir) or os.makedirs(image_dir)
        # 复制原图
        I = io.imread(osp.join(refer.IMAGE_DIR, img['file_name']))
        io.imsave(osp.join(image_dir, img['file_name']), I)
        

        cv2.imwrite(osp.join(image_dir, str(img_id)+".png"), image)

        # 将json格式的ref保存
        with open(osp.join(image_dir, str(img_id)+".json"), "w") as f:
            json.dump(ref, f)



        

    # i = 0
    # for ref_id in ref_ids:
    #     i += 1
    #     ref = refer.loadRefs(ref_id)[0]
    #     if len(ref['sentences']) < 2:
    #         continue

    #     print(ref)
    #     print('The label is %s.' % refer.Cats[ref['category_id']])
    #     plt.figure()
    #     # refer.getMask(ref)
    #     refer.showMask(ref)
        
    #     # refer.showRef(ref, seg_box='seg')
        
    #     plt.show()
    #     if i == 0:
    #         break
    #     # save
    #     plt.savefig('tmp.png')

        # plt.figure()
        # refer.showMask(ref)
        # plt.show()