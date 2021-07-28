import os
from lib.utils.snake import snake_coco_utils, snake_config, visualize_utils
import cv2
import numpy as np
import math
from lib.utils import data_utils
import torch.utils.data as data
from pycocotools.coco import COCO
from lib.config import cfg

# Loaded by the PyTorch dataloader
class Dataset(data.Dataset):
    def __init__(self, ann_file, data_root, split):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.split = split

        self.coco = COCO(ann_file)
        # Self.anns really contains a list of ImgIds
        self.anns = sorted(self.coco.getImgIds())
        # Keeps an image id if it contains annotations
        self.anns = np.array([ann for ann in self.anns if len(self.coco.getAnnIds(imgIds=ann, iscrowd=0))])
        self.anns = self.anns[:500] if split == 'mini' else self.anns
        self.json_category_id_to_contiguous_id = {v: i for i, v in enumerate(self.coco.getCatIds())}

    def process_info(self, img_id):
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=0)
        anno = self.coco.loadAnns(ann_ids)
        path = os.path.join(self.data_root, self.coco.loadImgs(int(img_id))[0]['file_name'])
        return anno, path, img_id

    def read_original_data(self, anno, path):
        img = cv2.imread(path)
        # Extract polygram annotations and converts it into a list containing np.arrays storing x,y
        # Syntax - instance_polys[object_number][0][polygram_index]
        instance_polys = [[np.array(poly).reshape(-1, 2) for poly in obj['segmentation']] for obj in anno]
        cls_ids = [self.json_category_id_to_contiguous_id[obj['category_id']] for obj in anno]
        return img, instance_polys, cls_ids

    # Convert poly annotations for flipping
    def transform_original_data(self, instance_polys, flipped, width, trans_output, inp_out_hw):
        output_h, output_w = inp_out_hw[2:]
        instance_polys_ = []
        for instance in instance_polys:
            # Create a deep copy of each poly in instance_polys
            polys = [poly.reshape(-1, 2) for poly in instance]

            # Flip annotations if necessary
            if flipped:
                polys_ = []
                for poly in polys:
                    # If flipped, then invert all the x co-ordinates
                    poly[:, 0] = width - np.array(poly[:, 0]) - 1 # Subtract by 1 because annotations are zero-index
                    polys_.append(poly.copy())
                polys = polys_

            # Perform affine transformation on annotations
            polys = snake_coco_utils.transform_polys(polys, trans_output, output_h, output_w)
            instance_polys_.append(polys)
        return instance_polys_

    def get_valid_polys(self, instance_polys, inp_out_hw):
        output_h, output_w = inp_out_hw[2:]
        instance_polys_ = []
        for instance in instance_polys:
            # Why restrict annotations to polygon of greater than 4?
            # Why use list comprehension if each instance only has one poly?
            instance = [poly for poly in instance if len(poly) >= 4]
            for poly in instance:
                # Clip x-coordinates to output's width
                poly[:, 0] = np.clip(poly[:, 0], 0, output_w - 1)
                # Clip y-cordinates with output height
                poly[:, 1] = np.clip(poly[:, 1], 0, output_h - 1)
            # Polygon must cover enough area
            polys = snake_coco_utils.filter_tiny_polys(instance)
            # Poly co-ordinates must be clock-wise
            polys = snake_coco_utils.get_cw_polys(polys)
            # Filter for unique co-ordinatesee
            polys = [poly[np.sort(np.unique(poly, axis=0, return_index=True)[1])] for poly in polys]
            # Must have 4 or more points
            polys = [poly for poly in polys if len(poly) >= 4]
            instance_polys_.append(polys)
        return instance_polys_

    def get_extreme_points(self, instance_polys):
        extreme_points = []
        for instance in instance_polys:
            points = [snake_coco_utils.get_extreme_points(poly) for poly in instance]
            extreme_points.append(points)
        return extreme_points

    def prepare_detection(self, box, poly, ct_hm, cls_id, wh, reg, ct_cls, ct_ind):
        ct_hm = ct_hm[cls_id]
        ct_cls.append(cls_id)

        x_min, y_min, x_max, y_max = box
        # Find the GT center point
        ct = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2], dtype=np.float32)
        ct_float = ct.copy()
        ct = np.round(ct).astype(np.int32)

        h, w = y_max - y_min, x_max - x_min
        radius = data_utils.gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        data_utils.draw_umich_gaussian(ct_hm, ct, radius)

        wh.append([w, h])
        ct_ind.append(ct[1] * ct_hm.shape[1] + ct[0])
        reg.append((ct_float  - ct).tolist())

        # Downscale annotation
        x_min, y_min = ct[0] - w / 2, ct[1] - h / 2
        x_max, y_max = ct[0] + w / 2, ct[1] + h / 2
        decode_box = [x_min, y_min, x_max, y_max]

        return decode_box

    def prepare_init(self, box, extreme_point, i_it_4pys, c_it_4pys, i_gt_4pys, c_gt_4pys, h, w):
        x_min, y_min = np.min(extreme_point[:, 0]), np.min(extreme_point[:, 1])
        x_max, y_max = np.max(extreme_point[:, 0]), np.max(extreme_point[:, 1])

        img_init_poly = snake_coco_utils.get_init(box)
        img_init_poly = snake_coco_utils.uniformsample(img_init_poly, snake_config.init_poly_num)
        can_init_poly = snake_coco_utils.img_poly_to_can_poly(img_init_poly, x_min, y_min, x_max, y_max)
        img_gt_poly = extreme_point
        can_gt_poly = snake_coco_utils.img_poly_to_can_poly(img_gt_poly, x_min, y_min, x_max, y_max)

        i_it_4pys.append(img_init_poly)
        c_it_4pys.append(can_init_poly)
        i_gt_4pys.append(img_gt_poly)
        c_gt_4pys.append(can_gt_poly)

    def prepare_evolution(self, poly, extreme_point, img_init_polys, can_init_polys, img_gt_polys, can_gt_polys):
        x_min, y_min = np.min(extreme_point[:, 0]), np.min(extreme_point[:, 1])
        x_max, y_max = np.max(extreme_point[:, 0]), np.max(extreme_point[:, 1])

        octagon = snake_coco_utils.get_octagon(extreme_point)
        img_init_poly = snake_coco_utils.uniformsample(octagon, snake_config.poly_num)
        can_init_poly = snake_coco_utils.img_poly_to_can_poly(img_init_poly, x_min, y_min, x_max, y_max)

        img_gt_poly = snake_coco_utils.uniformsample(poly, len(poly) * snake_config.gt_poly_num)
        tt_idx = np.argmin(np.power(img_gt_poly - img_init_poly[0], 2).sum(axis=1))
        img_gt_poly = np.roll(img_gt_poly, -tt_idx, axis=0)[::len(poly)]
        can_gt_poly = snake_coco_utils.img_poly_to_can_poly(img_gt_poly, x_min, y_min, x_max, y_max)

        img_init_polys.append(img_init_poly)
        can_init_polys.append(can_init_poly)
        img_gt_polys.append(img_gt_poly)
        can_gt_polys.append(can_gt_poly)

    def __getitem__(self, index):
        ann = self.anns[index]

        # anno - all the annotations associated with an image
        # path - file path to an image
        # img_id - the id of an image
        anno, path, img_id = self.process_info(ann)
        img, instance_polys, cls_ids = self.read_original_data(anno, path)

        height, width = img.shape[0], img.shape[1]
        orig_img, inp, trans_input, trans_output, flipped, center, scale, inp_out_hw = \
            snake_coco_utils.augment(
                img, self.split,
                snake_config.data_rng, snake_config.eig_val, snake_config.eig_vec,
                snake_config.mean, snake_config.std, instance_polys
            )

        # Fits annotations to augmentation techniques
        instance_polys = self.transform_original_data(instance_polys, flipped, width, trans_output, inp_out_hw)
        # Makes sure polygons are valid
        instance_polys = self.get_valid_polys(instance_polys, inp_out_hw)
        # Gets the extreme points
        extreme_points = self.get_extreme_points(instance_polys)

        # detection
        output_h, output_w = inp_out_hw[2:]
        ct_hm = np.zeros([cfg.heads.ct_hm, output_h, output_w], dtype=np.float32)
        wh = []
        reg = []
        ct_cls = []
        ct_ind = []

        # init
        i_it_4pys = []
        c_it_4pys = []
        i_gt_4pys = []
        c_gt_4pys = []

        # evolution
        i_it_pys = []
        c_it_pys = []
        i_gt_pys = []
        c_gt_pys = []

        for i in range(len(anno)):
            cls_id = cls_ids[i]
            instance_poly = instance_polys[i]
            instance_points = extreme_points[i]

            for j in range(len(instance_poly)):
                poly = instance_poly[j]
                extreme_point = instance_points[j]

                # Form a bbox from the annotation
                x_min, y_min = np.min(poly[:, 0]), np.min(poly[:, 1])
                x_max, y_max = np.max(poly[:, 0]), np.max(poly[:, 1])
                bbox = [x_min, y_min, x_max, y_max]
                h, w = y_max - y_min + 1, x_max - x_min + 1
                if h <= 1 or w <= 1:
                    continue

                decode_box = self.prepare_detection(bbox, poly, ct_hm, cls_id, wh, reg, ct_cls, ct_ind)
                self.prepare_init(decode_box, extreme_point, i_it_4pys, c_it_4pys, i_gt_4pys, c_gt_4pys, output_h, output_w)
                self.prepare_evolution(poly, extreme_point, i_it_pys, c_it_pys, i_gt_pys, c_gt_pys)

        ret = {'inp': inp}
        detection = {'ct_hm': ct_hm, 'radius': wh, 'reg': reg, 'ct_cls': ct_cls, 'ct_ind': ct_ind}
        init = {'i_it_4py': i_it_4pys, 'c_it_4py': c_it_4pys, 'i_gt_4py': i_gt_4pys, 'c_gt_4py': c_gt_4pys}
        evolution = {'i_it_py': i_it_pys, 'c_it_py': c_it_pys, 'i_gt_py': i_gt_pys, 'c_gt_py': c_gt_pys}
        ret.update(detection)
        ret.update(init)
        ret.update(evolution)
        # visualize_utils.visualize_snake_detection(orig_img, ret)
        # visualize_utils.visualize_snake_evolution(orig_img, ret)

        ct_num = len(ct_ind)
        meta = {'center': center, 'scale': scale, 'img_id': img_id, 'ann': ann, 'ct_num': ct_num}
        ret.update({'meta': meta})

        return ret

    def __len__(self):
        return len(self.anns)

