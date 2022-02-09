import os
import cv2
import json
import numpy as np
from lib.utils.snake import snake_config, snake_cityscapes_utils, snake_eval_utils, snake_poly_utils, visualize_utils
from external.cityscapesscripts.evaluation import evalInstanceLevelSemanticLabeling
import pycocotools.mask as mask_util
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import lib.utils.circle.kidpath_circle as kidpath_circle
from lib.utils.circle.circle_eval import CIRCLEeval
from lib.config import cfg
from lib.datasets.dataset_catalog import DatasetCatalog
from lib.utils import data_utils

debug = False

class Evaluator:
    def __init__(self, result_dir):
        self.results = []
        self.img_ids = []
        self.aps = []

        self.result_dir = result_dir
        os.system('mkdir -p {}'.format(self.result_dir))

        args = DatasetCatalog.get(cfg.test.dataset)
        self.ann_file = args['ann_file']
        self.data_root = args['data_root']
        self.coco = coco.COCO(self.ann_file)

        self.json_category_id_to_contiguous_id = {
            v: i for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.iter_num = 0

    def evaluate(self, output, batch):
        detection = output['detection']
        score = detection[:, 3].detach().cpu().numpy()
        label = detection[:, 4].detach().cpu().numpy().astype(int)
        py = output['py'][-1].detach().cpu().numpy() * snake_config.down_ratio

        if len(py) == 0:
            return

        img_id = int(batch['meta']['img_id'][0])
        center = batch['meta']['center'][0].detach().cpu().numpy()
        scale = batch['meta']['scale'][0].detach().cpu().numpy()

        h, w = batch['inp'].size(2), batch['inp'].size(3)
        trans_output_inv = data_utils.get_affine_transform(center, scale, 0, [w, h], inv=1)
        img = self.coco.loadImgs(img_id)[0]
        ori_h, ori_w = img['height'], img['width']
        py = [data_utils.affine_transform(py_, trans_output_inv) for py_ in py]
        rles = snake_eval_utils.coco_poly_to_rle(py, ori_h, ori_w)

        if cfg.debug_test:
            path = os.path.join(self.data_root, img["file_name"])
            orig_img = cv2.imread(path)

            for polys in py:
                poly_corrected = np.zeros(shape=(128, 2), dtype=np.int32)
                for i, (poly_x, poly_y) in enumerate(polys):
                    if poly_x < 0:
                        poly_x = 0
                    elif poly_x > ori_w:
                        poly_x = ori_w
                    if poly_y < 0:
                        poly_y = 0
                    elif poly_y > ori_h:
                        poly_y = ori_h

                    poly_corrected[i] = int(round(poly_x)), int(round(poly_y))
                cv2.polylines(orig_img, [np.int32(poly_corrected)], True, (0, 255, 0), 1)
            cv2.imshow("Prediction", orig_img)
            # cv2.imwrite("/home/sybbure/Documents/CircleSnake/data/debug/pred_contour.png", orig_img)
            cv2.waitKey(0)

        coco_dets = []
        for i in range(len(rles)):
            detection = {
                'image_id': img_id,
                'category_id': self.contiguous_category_id_to_json_id[label[i]],
                'segmentation': rles[i],
                'score': float('{:.2f}'.format(score[i]))
            }
            coco_dets.append(detection)

        self.results.extend(coco_dets)
        self.img_ids.append(img_id)

    def summarize(self):
        json.dump(self.results, open(os.path.join(self.result_dir, 'results.json'), 'w'))
        coco_dets = self.coco.loadRes(os.path.join(self.result_dir, 'results.json'))
        coco_eval = COCOeval(self.coco, coco_dets, 'segm')
        coco_eval.params.maxDets = [1000, 1000, 1000]
        coco_eval.params.imgIds = self.img_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        self.results = []
        self.img_ids = []
        self.aps.append(coco_eval.stats[0])
        return {'segm_ap': coco_eval.stats[0]}


class DetectionEvaluator:
    def __init__(self, result_dir):
        self.results = []
        self.img_ids = []
        self.aps = []
        self._valid_ids = [1]

        self.result_dir = result_dir
        os.system('mkdir -p {}'.format(self.result_dir))

        args = DatasetCatalog.get(cfg.test.dataset)
        self.ann_file = args['ann_file']
        self.data_root = args['data_root']
        # self.coco = coco.COCO(self.ann_file)
        self.circle = kidpath_circle.CIRCLE(self.ann_file)

        self.json_category_id_to_contiguous_id = {
            v: i for i, v in enumerate(self.circle.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.iter_num = 0

    def evaluate(self, output, batch):
        detection = output['detection']
        detection = detection[0] if detection.dim() == 3 else detection
        circle = detection[:, :3].detach().cpu().numpy() * snake_config.down_ratio
        score = detection[:, 3].detach().cpu().numpy()
        label = detection[:, 4].detach().cpu().numpy().astype(int)

        img_id = int(batch['meta']['img_id'][0])
        center = batch['meta']['center'][0].detach().cpu().numpy()
        scale = batch['meta']['scale'][0].detach().cpu().numpy()

        if len(circle) == 0:
            return

        h, w = batch['inp'].size(2), batch['inp'].size(3)
        trans_output_inv = data_utils.get_affine_transform(center, scale, 0, [w, h], inv=1)
        img = self.circle.loadImgs(img_id)[0]
        ori_h, ori_w = img['height'], img['width']


        if cfg.debug_test:
            # Read the image
            path = os.path.join(self.data_root, img["file_name"])
            pred_img = cv2.imread(path)

            # Overlay the prediction in green
            for i in range(len(label)):
                circle_ = data_utils.affine_transform(circle[i][:2].reshape(-1, 2), trans_output_inv).ravel()
                cv2.circle(pred_img, (int(circle_[0]), int(circle_[1])), int(circle[i][2]), (0, 255, 0), 2)

            # Show the image
            cv2.imshow("Prediction", pred_img)
            # cv2.imwrite(os.path.join("/home/ethan/Documents/CircleSnake/data/debug", str(self.iter_num) + ".png"), pred_img)
            self.iter_num += 1
            cv2.waitKey(0)

        circle_dets = []
        for i in range(len(label)):
            circle_ = data_utils.affine_transform(circle[i][:2].reshape(-1, 2), trans_output_inv).ravel()
            circle_ = list(map(lambda x: float('{:.2f}'.format(x)), circle_))
            detection = {
                'image_id': img_id,
                'category_id': self.contiguous_category_id_to_json_id[label[i]],
                'score': float('{:.2f}'.format(score[i])),
                'circle_center': [circle_[0], circle_[1]],
                'circle_radius': self._to_float(circle[i][2])
            }
            circle_dets.append(detection)

        self.results.extend(circle_dets)
        self.img_ids.append(img_id)

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_eval_circle_format(self, all_circles):
        detections = []
        for image_id in all_circles:
            for cls_ind in all_circles[image_id]:
                try:
                    category_id = self._valid_ids[cls_ind - 1]
                except:
                    aaa  =1
                for circle in all_circles[image_id][cls_ind]:
                    score = circle[3]
                    circle_out = list(map(self._to_float, circle[0:3]))

                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "score": float("{:.2f}".format(score)),
                        'circle_center': [circle_out[0], circle_out[1]],
                        'circle_radius': circle_out[2]
                    }
                    if len(circle) > 5:
                        extreme_points = list(map(self._to_float, circle[5:13]))
                        detection["extreme_points"] = extreme_points

                    # output_h = 512  # hard coded
                    # output_w = 512  # hard coded
                    # cp = [0, 0]
                    # cp[0] = circle_out[0]
                    # cp[1] = circle_out[1]
                    # cr = circle_out[2]
                    # if cp[0] - cr < 0 or cp[0] + cr > output_w:
                    #     continue
                    # if cp[1] - cr < 0 or cp[1] + cr > output_h:
                    #     continue

                    detections.append(detection)
        return detections

    def summarize(self):
        json.dump(self.results, open(os.path.join(self.result_dir, 'results.json'), 'w'))
        circle_dets = self.circle.loadRes(os.path.join(self.result_dir, 'results.json'))
        circle_eval = CIRCLEeval(self.circle, circle_dets, 'circle')
        circle_eval.params.imgIds = self.img_ids
        circle_eval.params.maxDets = [1000, 1000, 1000]
        circle_eval.evaluate()
        circle_eval.accumulate()
        circle_eval.summarize()
        self.results = []
        self.img_ids = []
        self.aps.append(circle_eval.stats[0])
        return {'det_ap': circle_eval.stats[0]}


Evaluator = Evaluator if cfg.segm_or_bbox == 'segm' else DetectionEvaluator
