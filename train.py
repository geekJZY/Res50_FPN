#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

import os.path as osp
import time
# import click
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from addict import Dict
from tqdm import tqdm
from utils.visualizer import Visualizer
from data.data_loading import *
from models.fpn import fpn
from utils.loss import lovasz_softmax, FocalLoss
from utils.metric import label_accuracy_hist, hist_to_score


def load_network(saveDir, network, network_label, epoch_label):
    save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
    save_path = osp.join(saveDir, save_filename)
    network.load_state_dict(torch.load(save_path))


def save_network(saveDir, network, network_label, epoch_label):
    save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
    save_path = osp.join(saveDir, save_filename)
    torch.save(network.to("cpu").state_dict(), save_path)
    if torch.cuda.is_available():
        network.cuda()


def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter, max_iter, power):
    if iter % lr_decay_iter or iter > max_iter:
        return None
    new_lr = init_lr * (1 - float(iter) / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def resize_target(target, size):
    new_target = np.zeros((target.shape[0], size, size), np.int32)
    for i, t in enumerate(target.numpy()):
        new_target[i, ...] = cv2.resize(t, (size,) * 2, interpolation=cv2.INTER_NEAREST)
    return new_target


def test(device, dataloader, model, logFile):
    torch.set_grad_enabled(False)
    model.eval()

    hist = np.zeros((7, 7))
    for i, (data, target) in enumerate(dataloader):
        # Image
        data = data.to(device)
        # Propagate forward
        output = model(data)
        outImg = cv2.resize(output[0].to("cpu").max(0)[1].numpy(), (target.shape[1],) * 2, interpolation=
        cv2.INTER_NEAREST)
        hist += label_accuracy_hist(target[0].to("cpu").numpy(), outImg, 7)

    _, acc_cls, recall_cls, iu, _ = hist_to_score(hist)
    print("accuracy of every class is {}, recall of every class is {}, iu of every class is {}".format(
        acc_cls, recall_cls, iu), file=open(logFile, "a"))
    print("mean iu is {}".format(np.nansum(iu[1:]) / 6), file=open(logFile, "a"))
    torch.set_grad_enabled(True)
    model.train()
    return np.nansum(iu[1:]) / 6


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
    x_test = torch.FloatTensor([])
    y_train = torch.FloatTensor([])
    y_vali = torch.FloatTensor([])

    # Dataset
    dataset = MultiDataSet(
        CONFIG.ROOT,
        CONFIG.CROPSIZE,
        CONFIG.INSIZE,
        preload=False
    )

    # DataLoader
    if CONFIG.RESAMPLEFLAG:
        batchSizeResample = CONFIG.BATCH_SIZE
        CONFIG.BATCH_SIZE = 1

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=CONFIG.BATCH_SIZE,
        num_workers=CONFIG.NUM_WORKERS,
        shuffle=True,
    )
    loader_iter = iter(loader)

    datasetTestVali = MultiDataSet(
        CONFIG.ROOT,
        CONFIG.CROPSIZE,
        CONFIG.INSIZE,
        phase="crossvali",
        testFlag=True,
        preload=False
    )

    # DataLoader
    loaderTestVali = torch.utils.data.DataLoader(
        dataset=datasetTestVali,
        batch_size=1,
        num_workers=CONFIG.NUM_WORKERS,
        shuffle=False,
    )

    datasetTestTrain = MultiDataSet(
        CONFIG.ROOT,
        CONFIG.CROPSIZE,
        CONFIG.INSIZE,
        phase="train",
        testFlag=True,
        preload=False
    )

    # DataLoader
    loaderTestTrain = torch.utils.data.DataLoader(
        dataset=datasetTestTrain,
        batch_size=1,
        num_workers=CONFIG.NUM_WORKERS,
        shuffle=False,
    )

    # Model
    model = fpn(CONFIG.N_CLASSES)

    # Optimizer
    optimizer = {
        "sgd": torch.optim.SGD(
            # cf lr_mult and decay_mult in train.prototxt
            params=[
                {
                    "params": model.parameters(),
                    "lr": CONFIG.LR,
                    "weight_decay": CONFIG.WEIGHT_DECAY,
                }
            ],
            momentum=CONFIG.MOMENTUM,
        )
    }.get(CONFIG.OPTIMIZER)

    # read old version
    model = nn.DataParallel(model)
    if CONFIG.ITER_START != 1:
        load_network(CONFIG.SAVE_DIR, model, "SateFPN", "latest")
        print("load previous model succeed, training start from iteration {}".format(CONFIG.ITER_START))
    model.to(device)

    # Loss definition
    criterion = lovasz_softmax
    criterion_focal = FocalLoss(device, gamma=6)

    #visualizer
    vis = Visualizer(CONFIG.DISPLAYPORT, CONFIG.EXPERIENT)

    model.train()
    iter_start_time = time.time()
    for iteration in range(CONFIG.ITER_START, CONFIG.ITER_MAX + 1):
        # Set a learning rate
        poly_lr_scheduler(
            optimizer=optimizer,
            init_lr=CONFIG.LR,
            iter=iteration - 1,
            lr_decay_iter=CONFIG.LR_DECAY,
            max_iter=CONFIG.ITER_MAX,
            power=CONFIG.POLY_POWER,
        )

        # Clear gradients (ready to accumulate)
        optimizer.zero_grad()

        iter_loss = 0
        for i in range(1, CONFIG.ITER_SIZE + 1):
            if not CONFIG.RESAMPLEFLAG:
                try:
                    data, target = next(loader_iter)
                except:
                    loader_iter = iter(loader)
                    data, target = next(loader_iter)
            else:
                cntFrame = 0
                clDataStart = time.time()
                clCnt = 0
                while cntFrame < batchSizeResample:
                    clCnt += 1
                    try:
                        dataOne, targetOne = next(loader_iter)
                    except:
                        loader_iter = iter(loader)
                        dataOne, targetOne = next(loader_iter)

                    hist = np.bincount(targetOne.numpy().flatten(), minlength=7)
                    hist = hist / np.sum(hist)
                    if np.nanmax(hist) <= 0.70:
                        if cntFrame == 0:
                            data = dataOne
                            target = targetOne
                        else:
                            data = torch.cat([data, dataOne])
                            target = torch.cat([target, targetOne])
                        cntFrame += 1
                print("collate data takes %.2f sec, collect %d time" % (time.time() - clDataStart, clCnt))

            # Image
            data = data.to(device)

            # Propagate forward
            output = model(data)
            # Loss
            loss = 0
            # Resize target for {100%, 75%, 50%, Max} outputs
            target_ = resize_target(target, output.size(2))
            # classmap = class_to_target(target_, CONFIG.N_CLASSES)
            # target_ = label_bluring(classmap)  # soft crossEntropy target
            target_ = torch.from_numpy(target_).long()
            target_ = target_.to(device)
            # Compute crossentropy loss
            if CONFIG.CENTERCOMPARE:
                loss += criterion(output[:,:,60:-60,60:-60], target_[:,60:-60,60:-60])
            else:
                loss += criterion(output, target_) + 0.5 * criterion_focal(output, target_)
            # Backpropagate (just compute gradients wrt the loss)
            loss /= float(CONFIG.ITER_SIZE)
            loss.backward()

            iter_loss += float(loss)

        # Update weights with accumulated gradients
        optimizer.step()
        # Visualizer and Summery Writer
        if iteration % CONFIG.ITER_TF == 0:
            print("itr {}, loss is {}".format(iteration, iter_loss), file=open(CONFIG.LOGNAME, "a"))  #
            # print("time taken for each iter is %.3f" % ((time.time() - iter_start_time)/iteration))

        if iteration % 5 == 0:
            vis.drawLine(torch.FloatTensor([iteration]), torch.FloatTensor([iter_loss]))
            vis.displayImg(inputImgTransBack(data), classToRGB(output[0].to("cpu").max(0)[1]),
                           classToRGB(target_[0].to("cpu")))
        # Save a model
        if iteration % CONFIG.ITER_SNAP == 0:
            save_network(CONFIG.SAVE_DIR, model, "SateFPN", iteration)

        # Save a model
        if iteration % 200 == 0:
            save_network(CONFIG.SAVE_DIR, model, "SateFPN", "latest")

        # test a model
        if (iteration + 1) % 5000 == 0:
            x_test = torch.cat((x_test, torch.FloatTensor([iteration])))
            print("test in trainset", file=open(CONFIG.LOGNAME, "a"))
            mIOU = test(device, loaderTestTrain, model, CONFIG.LOGNAME)
            y_train = torch.cat((y_train, torch.FloatTensor([mIOU])))
            print("test in validationset", file=open(CONFIG.LOGNAME, "a"))
            mIOU = test(device, loaderTestVali, model, CONFIG.LOGNAME)
            y_vali = torch.cat((y_vali, torch.FloatTensor([mIOU])))
            vis.drawTestLine(x_test, y_vali, y_train)

if __name__ == "__main__":
    main()
