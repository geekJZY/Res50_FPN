#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

import os.path as osp

import cv2
import torch
import torch.nn as nn
import yaml
from addict import Dict
from data.data_loading import *
from models.fpn import fpn
# from models.fcn import FCN8
# from utils.visualizer import Visualizer
from utils.metric import label_accuracy_hist, hist_to_score


def load_network(saveDir, network, network_label, epoch_label):
    save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
    save_path = osp.join(saveDir, save_filename)
    network.load_state_dict(torch.load(save_path))
    print("the network load is in " + save_path, file=open("output.txt", "a"))


def save_network(saveDir, network, network_label, epoch_label):
    save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
    save_path = osp.join(saveDir, save_filename)
    torch.save(network.to("cpu").state_dict(), save_path)
    if torch.cuda.is_available():
        network.cuda()


def main():
    config = "config/cocostuff.yaml"
    cuda = True
    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")

    if cuda:
        current_device = torch.cuda.current_device()
        print("Running on", torch.cuda.get_device_name(current_device))
    else:
        print("Running on CPU")

    # Configuration
    CONFIG = Dict(yaml.load(open(config)))
    CONFIG.SAVE_DIR = osp.join(CONFIG.SAVE_DIR, CONFIG.EXPERIENT)
    CONFIG.LOGNAME = osp.join(CONFIG.SAVE_DIR, "log.txt")

    # Model
    torch.set_grad_enabled(False)
    model = fpn(CONFIG.N_CLASSES)
    model = nn.DataParallel(model)

    load_network(CONFIG.SAVE_DIR, model, "SateFPN", "latest")
    model.to(device)
    model.eval()

    # Dataset
    dataset = MultiDataSet(
        CONFIG.ROOT,
        CONFIG.CROPSIZE,
        CONFIG.INSIZE,
        phase="offical_crossvali",
        testFlag=True,
        final=True,
        preload=False
    )

    # DataLoader
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=CONFIG.NUM_WORKERS,
        shuffle=False,
    )

    for i, (data, name) in enumerate(loader):
        print("Process img%d" % i)
        # Image
        data = data.to(device)

        # Propagate forward
        output = model(data)

        # Resize target for {100%, 75%, 50%, Max} outputs
        outImg = cv2.resize(output[0].to("cpu").max(0)[1].numpy(), (2448,) * 2, interpolation=
                            cv2.INTER_NEAREST)
        print(classToRGB(outImg, outformat="image").shape)
        print(osp.join(CONFIG.ROOT, "offical_crossvali", "Label", name[0]))
        cv2.imwrite(osp.join(CONFIG.ROOT, "offical_crossvali", "Label", name[0]), cv2.cvtColor(classToRGB(outImg, outformat="image"),
                                                                                  cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()