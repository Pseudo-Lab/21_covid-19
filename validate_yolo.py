import sys

sys.path.append('yolov5')

import os
import glob
import argparse
from pathlib import Path

import yaml

from yolov5 import test
from models.experimental import attempt_load
from yolo.datasets import create_dataloader
from utils.general import check_img_size, check_file, colorstr


def _validate(weight_path, save_dir, validation_fold, opt):
    save_dir = Path(save_dir)
    batch_size = opt.batch_size
    single_cls = opt.single_cls
    data = opt.data
    workers = opt.workers

    root = opt.root
    datatype = opt.datatype
    imgsz = opt.img_size

    model = attempt_load(weight_path, map_location='cuda:0')

    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(imgsz, s=gs)  # check image size

    testloader, _ = create_dataloader(root, datatype, False, validation_fold, imgsz, batch_size * 2,
                                      gs, opt, hyp=None, cache=False, rect=True,
                                      rank=-1, workers=workers, world_size=1, pad=0.5,
                                      prefix=colorstr('valid: '))

    with open(data) as f:
        data_dict = yaml.safe_load(f)  # data dict

    _ = test.run(data_dict,
                 batch_size=batch_size * 2,
                 imgsz=imgsz,
                 model=model,
                 single_cls=single_cls,
                 dataloader=testloader,
                 save_dir=save_dir,
                 verbose=True,
                 plots=False,
                 save_txt=True,
                 save_conf=True,
                 wandb_logger=None,
                 compute_loss=None)


def validate(opt):
    path_format = os.path.join(opt.exp, '*', 'weights', 'best.pt')
    weight_paths = glob.glob(path_format)
    for weight_path in weight_paths:
        opt_path = os.path.join(os.path.dirname(weight_path), '..', 'opt.yaml')
        with open(opt_path) as f:
            opt_train = yaml.safe_load(f)
        validation_fold = opt_train['validation_fold']

        # save_dir = os.path.join(opt.exp, 'result', f'fold_{validation_fold}')
        save_dir = opt.exp  # this is CV - results will be OOF for each model
        os.makedirs(os.path.join(save_dir, 'labels'), exist_ok=True)    # test.run attaches labels to save path
        _validate(weight_path, save_dir, validation_fold, opt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp', type=str, help='path to experiment results')
    parser.add_argument('--data', type=str, default='yolo/siim.yaml', help='dataset.yaml path')
    parser.add_argument('--datatype', type=str, default='px1280')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    opt = parser.parse_args()

    # hard-code
    opt.single_cls = True
    opt.save_txt = True
    opt.save_conf = True
    opt.data = check_file(opt.data)  # check file

    project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    project_root = os.path.abspath(project_root)
    data_root = os.path.join(project_root, 'data')
    opt.root = data_root

    validate(opt)
