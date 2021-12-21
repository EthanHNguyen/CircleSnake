import os
from lib.utils.snake import snake_coco_utils, snake_config, visualize_utils
import cv2
import numpy as np
import math
from lib.utils import data_utils
import torch.utils.data as data
from pycocotools.coco import COCO
from lib.config import cfg
import math

debug = False

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
        # Extract polygon annotations and converts it into a list containing np.arrays storing x,y
        # Syntax - gt_circles[object_number][0][polygram_index]
        instance_polys = [[np.array(poly).reshape(-1, 2) for poly in obj['segmentation']] for obj in anno]
        cls_ids = [self.json_category_id_to_contiguous_id[obj['category_id']] for obj in anno]
        return img, instance_polys, cls_ids

    def read_original_circle_data(self, anno, path):
        """
        This function returns the associated ground truth bounding circle annotation for each object in the image.

        Args:
            anno: The annotation file associated to an image containing >= 1 bounding circles.
            path: The file path to an image.

        Returns:
            img: The image in an cv2 object.
            GT_circle_annotations: A list containing only the bounding circle and category id.
        """
        img = cv2.imread(path)
        gt_circles = []
        for ann in anno:
            gt_circles.append({
                "circle_center": ann["circle_center"],
                "circle_radius": ann["circle_radius"]
                # "class_id" : ann["category_id"]
            })
        instance_polys = [[np.array(poly).reshape(-1, 2) for poly in obj['segmentation']] for obj in anno]

        cls_ids = [self.json_category_id_to_contiguous_id[obj['category_id']] for obj in anno]
        return img, instance_polys, gt_circles, cls_ids

    # Convert gt_circle annotations for flipping
    def transform_original_circle_data(self, instance_polys, gt_circles, flipped, width, trans_output, inp_out_hw):
        output_h, output_w = inp_out_hw[2:]
        gt_circles_ = []
        # for circle in gt_circles:
        #     circle = circle.copy()
        #     # Flip annotations if necessary
        #     if flipped:
        #         circle["circle_center"][0] = width - circle["circle_center"][0] - 1
        #
        #     # Perform affine transformation on annotations
        #     circle["circle_center"] = snake_coco_utils.affine_transform_point(circle["circle_center"], trans_output)
        #     if (0 <= circle["circle_center"][0] < output_w - 1 and 0 <= circle["circle_center"][1] < output_h - 1):
        #         circle["circle_radius"] /= snake_config.down_ratio
        #         gt_circles_.append(circle.copy())

        # # Treat center point as a polygon
        # # TODO - clean-up
        # input = [np.array([circle["circle_center"], [1,1], [2,2]])]
        # output = snake_coco_utils.transform_polys(input, trans_output, output_h, output_w)
        # if output:
        #     circle["circle_center"] = [output[0][0][0], output[0][0][1]]
        #
        #     # Make sure circle_center is in bounds
        #     x, y = circle["circle_center"]
        #     assert(0 <= x < output_w)
        #     assert(0 <= y < output_h)
        #
        #     gt_circles_.append(circle.copy())

        instance_polys_ = []
        for instance in instance_polys:
            polys = [poly.reshape(-1, 2) for poly in instance]

            if flipped:
                polys_ = []
                for poly in polys:
                    poly[:, 0] = width - np.array(poly[:, 0]) - 1
                    polys_.append(poly.copy())
                polys = polys_

            polys = snake_coco_utils.transform_polys(polys, trans_output, output_h, output_w)
            instance_polys_.append(polys)
        return gt_circles_, instance_polys_

    def get_valid_polys(self, instance_polys, inp_out_hw):
        output_h, output_w = inp_out_hw[2:]
        instance_polys_ = []
        for instance in instance_polys:
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
            # Filter for unique co-ordinates
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

    def prepare_detection(self, poly, ct_hm, cls_id, retCenter, retRadius, reg, ct_cls, ct_ind):
        """
        Prepares the heatmaps for detection

        Args:
            box: The bounding box
            gt_circle: The polygon annotations
            ct_hm: A heatmap containing the center point
            cls_id: The class id of the associated object
            radius: The radius of the object
            reg: The offset of the object
            ct_cls: Same as the class id.
            ct_ind: Where the center point is one the image. row * row_size + col

        Returns: The downscaled bounding box

        """

        ct_hm = ct_hm[cls_id]
        ct_cls.append(cls_id)

        # Find circle from poly
        x, y, radius = snake_coco_utils.numerical_stable_circle(poly)
        # x, y, radius = int(round(x)), int(round(y)), int(math.ceil(radius))
        x, y, radius = int(round(x)), int(round(y)), int(round(radius))

        ct = [x, y]
        ct_float = ct.copy()
        ct = np.round(ct).astype(np.int32)
        data_utils.draw_umich_gaussian(ct_hm, ct, radius)
        retRadius.append([radius])

        # assert(0 <= x <= 128)
        # assert (0 <= y <= 128)
        retCenter.append([x, y])

        # assert (0 <= ct[1] * ct_hm.shape[1] + ct[0] < (128 ** 2))
        ct_ind.append(ct[1] * ct_hm.shape[1] + ct[0])
        reg.append((ct_float - ct).tolist())

        # Downscale annotation
        # x_min, y_min = ct[0] - w / 2, ct[1] - h / 2
        # x_max, y_max = ct[0] + w / 2, ct[1] + h / 2
        # decode_box = [x_min, y_min, x_max, y_max]
        #
        # return decode_box

        return {
            "circle_center": [x, y],
            "circle_radius": radius
        }

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

    def prepare_evolution(self, poly, gt_circle, img_init_polys, can_init_polys, img_gt_polys, can_gt_polys):
        x_min, y_min = np.min(poly[:, 0]), np.min(poly[:, 1])
        x_max, y_max = np.max(poly[:, 0]), np.max(poly[:, 1])

        # octagon = snake_coco_utils.get_octagon(extreme_point)
        img_init_poly = snake_coco_utils.uniformsample_circle(gt_circle, snake_config.poly_num)
        can_init_poly = snake_coco_utils.img_poly_to_can_poly(img_init_poly, x_min, y_min, x_max, y_max)

        img_gt_poly = snake_coco_utils.uniformsample(poly, len(poly) * snake_config.gt_poly_num)
        tt_idx = np.argmin(np.power(img_gt_poly - img_init_poly[0], 2).sum(axis=1))
        img_gt_poly = np.roll(img_gt_poly, -tt_idx, axis=0)[::len(poly)]
        can_gt_poly = snake_coco_utils.img_poly_to_can_poly(img_gt_poly, x_min, y_min, x_max, y_max)

        img_init_polys.append(img_init_poly)
        can_init_polys.append(can_init_poly)
        img_gt_polys.append(img_gt_poly)
        can_gt_polys.append(can_gt_poly)

    # Given an image id, fetch the associated annotations and format it nicely
    def __getitem__(self, index):
        ann = self.anns[index]

        anno, path, img_id = self.process_info(ann)
        img, instance_polys, gt_circles, cls_ids = self.read_original_circle_data(anno, path)

        height, width = img.shape[0], img.shape[1]

        orig_img, inp, trans_input, trans_output, flipped, center, scale, inp_out_hw = \
            snake_coco_utils.augment_circle(
                img, self.split,
                snake_config.data_rng, snake_config.eig_val, snake_config.eig_vec,
                snake_config.mean, snake_config.std, instance_polys
            )

        # Fits annotations to augmentation techniques
        gt_circles, instance_polys = self.transform_original_circle_data(instance_polys, gt_circles, flipped, width,
                                                                         trans_output, inp_out_hw)

        # Makes sure polygons are valid
        instance_polys = self.get_valid_polys(instance_polys, inp_out_hw)

        # Gets the extreme points
        # extreme_points = self.get_extreme_points(GT_circle_annotations)

        # detection
        output_h, output_w = inp_out_hw[2:]
        ct_hm = np.zeros([cfg.heads.ct_hm, output_h, output_w], dtype=np.float32)
        circle_center = []
        radius = []
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
            # gt_circle = gt_circles[i]
            instance_poly = instance_polys[i]

            # index into object
            for j in range(len(instance_poly)):
                poly = instance_poly[j]

                # Form a bbox from the annotation
                x_min, y_min = np.min(poly[:, 0]), np.min(poly[:, 1])
                x_max, y_max = np.max(poly[:, 0]), np.max(poly[:, 1])
                bbox = [x_min, y_min, x_max, y_max]
                h, w = y_max - y_min + 1, x_max - x_min + 1
                if h <= 1 or w <= 1:
                    continue

                gt_circle = self.prepare_detection(poly, ct_hm, cls_id, circle_center, radius, reg, ct_cls, ct_ind)
                # gt_circle = self.prepare_detection(poly, gt_circle, ct_hm, cls_id, radius, center, reg, ct_cls, ct_ind)
                # self.prepare_init(decode_box, extreme_point, i_it_4pys, c_it_4pys, i_gt_4pys, c_gt_4pys, output_h, output_w)
                self.prepare_evolution(poly, gt_circle, i_it_pys, c_it_pys, i_gt_pys, c_gt_pys)

        ret = {'inp': inp}
        detection = {'ct_hm': ct_hm, 'center': center, 'circle_center': circle_center, 'radius': radius,
                     'reg': reg, 'ct_cls': ct_cls, 'ct_ind': ct_ind}
        # init = {'i_it_4py': i_it_4pys, 'c_it_4py': c_it_4pys, 'i_gt_4py': i_gt_4pys, 'c_gt_4py': c_gt_4pys}
        evolution = {'i_it_py': i_it_pys, 'c_it_py': c_it_pys, 'i_gt_py': i_gt_pys, 'c_gt_py': c_gt_pys}
        ret.update(detection)
        # ret.update(init)
        ret.update(evolution)

        if debug:
            # Visualize the ground truth of CircleNet
            visualize_utils.visualize_snake_detection_circle(orig_img, ret)

            # Visualize the initial contour
            visualize_utils.visualize_snake_evolution(orig_img, ret)

        ct_num = len(ct_ind)
        meta = {'center': center, 'scale': scale, 'img_id': img_id, 'ann': ann, 'ct_num': ct_num}
        ret.update({'meta': meta})

        return ret

    def __len__(self):
        return len(self.anns)
