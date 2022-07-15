import os
import os.path

import cv2
import numpy as np
import pycocotools.coco as coco
import torch
import tqdm
from custom_evaluator import Evaluator
from pycocotools.cocoeval import COCOeval

from lib.config import args, cfg
from lib.datasets import make_data_loader
from lib.evaluators import make_evaluator
from lib.networks import make_network
from lib.utils.circle.circle_eval import CIRCLEeval
from lib.utils.net_utils import load_network

DATA = "/home/ethan/Documents/CircleSnake/data/kidpath_multiROI/test2022"
GT_ANN = "/home/ethan/Documents/CircleSnake/data/kidpath_multiROI/kidneypath_test2022.json"
PRED_ANNS = [["CircleSnake",
              "/home/ethan/Documents/CircleSnake/data/result/circle_snake/CircleSnake_glomeruli_v24/results.json"],
             ["Mask R-CNN",
              "/home/ethan/Documents/detectron2/projects/MoNuSeg/glomeruli/coco_instances_results.json"
              ]]
OUPUT_DIR = "/home/ethan/Documents/CircleSnake/debug_metrics"

METRIC = ["Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] =",
          "Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] =",
          "Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] =",
          "Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] =",
          "Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] =",
          "Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] =",
          "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] =",
          "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] =",
          "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] =",
          "Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] =",
          "Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] =",
          "Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] ="]

if __name__ == "__main__":
    coco = coco.COCO(GT_ANN)

    for pred_ann in PRED_ANNS:
        coco_dets = coco.loadRes(pred_ann[1])

        for imgId in coco.getImgIds():
            output_dir = os.path.join(OUPUT_DIR, str(imgId))

            if not os.path.exists(output_dir):
                os.mkdir(output_dir)

            retStr = pred_ann[0] + "\n"

            # COCO AP score
            coco_eval = COCOeval(coco, coco_dets, 'segm')
            coco_eval.params.imgIds = [imgId]
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            for idx, stat in enumerate(coco_eval.stats):
                retStr = retStr + METRIC[idx] + str(stat) + "\n"

            # Dice score
            def getMask(anns):
                instance_masks = [coco.annToMask(ann) for ann in anns]
                final_mask = np.zeros([512, 512]).astype(np.bool)
                for mask in instance_masks:
                    final_mask = np.logical_or(mask.astype(np.bool), final_mask)
                return final_mask, instance_masks


            gt_ann = coco.imgToAnns[imgId]
            gt_mask, gt_instance_masks = getMask(gt_ann)
            # cv2.imshow("GT", gt_mask.astype(np.uint8) * 255)

            pred_ann_RLE = coco_dets.imgToAnns[imgId]
            pred_mask, pred_instance_masks = getMask(pred_ann_RLE)
            pred_mask_score = [x["score"] for x in pred_ann_RLE]
            # cv2.imshow("PRED", pred_mask.astype(np.uint8) * 255)
            # cv2.waitKey(0)

            intersection = np.logical_and(gt_mask, pred_mask)
            dice_score = 2 * intersection.sum() / (gt_mask.sum() + pred_mask.sum())

            # print(dice_score)
            retStr += "\n" + str(dice_score) + "\n\n"

            output_text_path = os.path.join(output_dir, pred_ann[0] + "_metric.txt")
            with open(output_text_path, 'w') as f:
                f.write(retStr)

            img_file_name = coco.loadImgs(imgId)[0]["file_name"]
            img_path = os.path.join(DATA, img_file_name)
            img = cv2.imread(img_path)
            # cv2.imshow("img",  img)
            # cv2.waitKey(0)

            # pred_contours = []
            pred_img = img.copy()
            # for mask in pred_instance_masks:
            #     cv2.imshow("pred", mask.astype(np.uint8) * 255)
            #     cv2.waitKey(0)
            pred_contours_ = [cv2.findContours(np.ascontiguousarray(mask * 255, np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1] for mask in pred_instance_masks]
            for idx, contour in enumerate(pred_contours_):
                cv2.polylines(pred_img, contour, True, (0, 255, 0), 2)
                cv2.rectangle(pred_img, (int(contour[0][0][0][0]), int(contour[0][0][0][1])), (int(contour[0][0][0][0]) + 40, int(contour[0][0][0][1]) - 15), (0,0,0), -1)
                cv2.putText(pred_img, str(round(pred_mask_score[idx], 2)), (int(contour[0][0][0][0]), int(contour[0][0][0][1])), 0, 0.5, (0,255,0), 1)

            # cv2.imshow("pred", pred_img)
            # cv2.waitKey(0)
            pred_img_path = os.path.join(output_dir, pred_ann[0] + ".png")
            cv2.imwrite(pred_img_path, pred_img)

            gt_img = img.copy()
            gt_contours = cv2.findContours(gt_mask.astype(np.uint8) * 255, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
            for contour in gt_contours:
                cv2.polylines(gt_img, [contour], True, (0, 255, 0), 2)
            # cv2.imshow("gt", gt_img)
            # cv2.waitKey(0)
            gt_img_path = os.path.join(output_dir, "truth.png")
            cv2.imwrite(gt_img_path, gt_img)










