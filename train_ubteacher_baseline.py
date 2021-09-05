#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import sys
sys.path.append('unbiased-teacher')

import argparse
import os

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch

# hacky way to register
from ubteacher.modeling.meta_arch.rcnn import TwoStagePseudoLabGeneralizedRCNN
from ubteacher.modeling.proposal_generator.rpn import PseudoLabRPN
from ubteacher.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab
import ubteacher.data.datasets.builtin

from ubteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel

from misc.config import add_siim_base_config, add_baseline_config, add_ubteacher_config
from misc.initialization import set_metadata_catalog, set_data_catalog
from runner.trainer import SIIMBaselineTrainer
from runner.ubteacher_trainer import UBTeacherTrainer


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_ubteacher_config(cfg)    # need to be first
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    add_siim_base_config(cfg, args)
    set_metadata_catalog(args)
    set_data_catalog(cfg)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    if cfg.SEMISUPNET.Trainer == "ubteacher":
        Trainer = UBTeacherTrainer  # TODO;
    elif cfg.SEMISUPNET.Trainer == "baseline":
        Trainer = SIIMBaselineTrainer
    else:
        raise ValueError("Trainer Name is not found.")

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    return trainer.train()


if __name__ == "__main__":
    siim_parser = argparse.ArgumentParser()
    dataroot = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    dataroot = os.path.abspath(dataroot)

    siim_parser.add_argument('--dataroot', type=str, default=dataroot)
    siim_parser.add_argument('--datatype', type=str, default='px1280')
    siim_parser.add_argument('--validation-fold', type=int, default=0)
    siim_parser.add_argument('--no-one-class', dest='one_class', action='store_false', default=True)
    siim_parser.add_argument('--use-healthy', action='store_true', default=False)
    siim_parser.add_argument('--use-empty', action='store_true', default=False)
    
    siim_args, unknown = siim_parser.parse_known_args()

    args = default_argument_parser().parse_args(unknown, namespace=siim_args)

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
