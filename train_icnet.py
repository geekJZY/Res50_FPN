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
from models.icnet import icnet
from utils.loss import lovasz_softmax
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


def test(device, dataloader, model, logFile, num_class=7, cropsize=513):
    torch.set_grad_enabled(False)
    model.train()

    hist = np.zeros((num_class, num_class))
    for i, (data, target) in enumerate(dataloader):
        print("process image {}".format(i))
        # Image
        data = data.to(device)
        b, _, h, w = data.size()
        print(data.size())
        output = torch.zeros([b, num_class, h, w], dtype=torch.float32, device=device)
        for patchX in range(0, h - 1, cropsize):
            for patchY in range(0, w - 1, cropsize):
                # Propagate forward
                if patchX >= h - cropsize:
                    patchX = h - cropsize
                if patchY >= w - cropsize:
                    patchY = w - cropsize

                outputCrop = model(data[:, :, patchX:patchX + cropsize, patchY:patchY + cropsize])
                output[:, :, patchX: (patchX + cropsize), patchY: (patchY + cropsize)] = \
                    F.interpolate(outputCrop[2], size=(513, 513), mode='bilinear')
        outImg = output[0].max(0)[1].to("cpu").numpy()
        hist += label_accuracy_hist(target[0].to("cpu").numpy(), outImg, num_class)
        _, _, _, iu, _ = hist_to_score(hist)
        print("mean iu is {}".format(np.nansum(iu[1:]) / 6))

    _, acc_cls, recall_cls, iu, _ = hist_to_score(hist)
    print("accuracy of every class is {}, recall of every class is {}, iu of every class is {}".format(
        acc_cls, recall_cls, iu), file=open(logFile, "a"))
    print("mean iu is {}".format(np.nansum(iu[1:]) / 6), file=open(logFile, "a"))
    torch.set_grad_enabled(True)
    model.train()
    return np.nansum(iu[1:]) / 6


def global2patch(images, masks, cropSize, batchSizeSmall):
    '''
    input:
    images(tensor):b,c,h,w
    masks(tensor):b,h,w
    cropSize(int): the size of crop patch
    output:
    imageCrops(tensor): batchSize,c,(size)
    maskCrops(tensor): batchSize,c,(size)
    bbox: list [b* nparray(x1, y1, x2, y2)] the (x1,y1) is the left_top of bbox, (x2, y2) is the right_bottom of bbox
    there are in range [0, 1]. x is corresponding to width dimension and y is corresponding to height dimension
    '''
    b, c, h, w = images.size()

    imageCrops = []
    maskCrops = []
    bbox = []
    for cntImg in range(b):
        bboxTemp = []
        w_offset = np.random.randint(0, max(0, w - cropSize - 1), size=batchSizeSmall)
        h_offset = np.random.randint(0, max(0, h - cropSize - 1), size=batchSizeSmall)
        for cntCrop in range(batchSizeSmall):
            imageCrops.append(images[cntImg, :, h_offset[cntCrop]:h_offset[cntCrop]+cropSize,
                              w_offset[cntCrop]:w_offset[cntCrop]+cropSize])
            maskCrops.append(masks[cntImg, h_offset[cntCrop]:h_offset[cntCrop]+cropSize,
                             w_offset[cntCrop]:w_offset[cntCrop]+cropSize])
            bboxTemp.append(np.array([w_offset[cntCrop]/(w-1), h_offset[cntCrop]/(h-1),
                                      (w_offset[cntCrop] + cropSize)/(w-1), (h_offset[cntCrop] + cropSize)/(h-1)]))
        bbox.append(bboxTemp)

    return torch.stack(imageCrops, dim=0), torch.stack(maskCrops, dim=0), bbox


def main():
    config = "config/config_icnet.yaml"
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
    model = icnet(n_classes=CONFIG.N_CLASSES)

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

    if CONFIG.ITER_START != 1:
        load_network(CONFIG.SAVE_DIR, model, "SateFPN", "latest")
        print("load previous model succeed, training start from iteration {}".format(CONFIG.ITER_START))
    model.to(device)

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
            try:
                data, target = next(loader_iter)
            except:
                loader_iter = iter(loader)
                data, target = next(loader_iter)

            # Image
            cntFrame = 0
            dataResample = []
            targetResample = []
            data = data
            data, target, _ = global2patch(data, target, CONFIG.CROPSIZE, 5)
            sampleBar = 0.70 + iteration/30000
            if sampleBar < 1:
                while cntFrame < 10:
                    for cntBatch in range(target.shape[0]):
                        hist = np.bincount(target[cntBatch].numpy().flatten(), minlength=7)
                        hist = hist / np.sum(hist)
                        if np.nanmax(hist) <= sampleBar:
                            dataResample.append(data[cntBatch])
                            targetResample.append(target[cntBatch])
                            cntFrame += 1
                    if cntFrame < 10:
                        try:
                            data, target = next(loader_iter)
                        except:
                            loader_iter = iter(loader)
                            data, target = next(loader_iter)
                        data, target, _ = global2patch(data, target, CONFIG.CROPSIZE, 3)
                data, target = torch.stack(dataResample, dim=0), torch.stack(targetResample, dim=0)
            data = data.to(device)
            # Propagate forward
            output = model(data)
            # Loss
            loss = 0
            # classmap = class_to_target(target_, CONFIG.N_CLASSES)
            # target_ = label_bluring(classmap)  # soft crossEntropy target
            target_ = target.long()
            target_ = target_.to(device)
            # Compute crossentropy loss
            loss += model.loss(output, target_)
            # Backpropagate (just compute gradients wrt the loss)
            loss /= float(CONFIG.ITER_SIZE)
            loss.backward()

            iter_loss += float(loss)

        # Update weights with accumulated gradients
        optimizer.step()
        # Visualizer and Summery Writer
        if iteration % CONFIG.ITER_TF == 0:
            print("itr {}, loss is {}".format(iteration, iter_loss), file=open(CONFIG.LOGNAME, "a"))  #
            print("time taken for each iter is %.3f" % ((time.time() - iter_start_time)/(iteration-CONFIG.ITER_START+1)))

        if iteration % 5 == 0:
            vis.drawLine(torch.FloatTensor([iteration]), torch.FloatTensor([iter_loss]))
            vis.displayImg(inputImgTransBack(data), classToRGB(output[2][0].to("cpu").max(0)[1]),
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
