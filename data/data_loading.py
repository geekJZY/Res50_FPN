# Prepare Dataset
from os import listdir
from os.path import join
import torch
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
import random
from torchvision.transforms import ToTensor
import cv2
import math
import time
import scipy.io

ImageFile.LOAD_TRUNCATED_IMAGES = True


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def find_label_map_name(img_filenames, labelExtension=".png"):
    img_filenames = img_filenames.replace('_sat.jpg', '_mask')
    return img_filenames + labelExtension


def RGB_mapping_to_class(label):
    l, w = label.shape[0], label.shape[1]
    classmap = np.zeros(shape=(l, w))
    indices = np.where(np.all(label == (0, 255, 255), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 1
    indices = np.where(np.all(label == (255, 255, 0), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 2
    indices = np.where(np.all(label == (255, 0, 255), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 3
    indices = np.where(np.all(label == (0, 255, 0), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 4
    indices = np.where(np.all(label == (0, 0, 255), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 5
    indices = np.where(np.all(label == (255, 255, 255), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 6
    indices = np.where(np.all(label == (0, 0, 0), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 0
    #     plt.imshow(colmap)
    #     plt.show()
    return classmap


def classToRGB(label, outformat="tensor"):
    l, w = label.shape[0], label.shape[1]
    colmap = np.zeros(shape=(l, w, 3))
    indices = np.where(label == 1)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [0, 255, 255]
    indices = np.where(label == 2)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [255, 255, 0]
    indices = np.where(label == 3)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [255, 0, 255]
    indices = np.where(label == 4)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [0, 255, 0]
    indices = np.where(label == 5)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [0, 0, 255]
    indices = np.where(label == 6)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [255, 255, 255]
    indices = np.where(label == 0)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [0, 0, 0]
    transform = ToTensor();
    #     plt.imshow(colmap)
    #     plt.show()
    if outformat == "tensor":
        return transform(colmap)
    else:
        return  colmap.astype(np.uint8)


def class_to_target(inputs, numClass):
    batchSize, l, w = inputs.shape[0], inputs.shape[1], inputs.shape[2]
    target = np.zeros(shape=(batchSize, l, w, numClass), dtype=np.float32)
    for index in range(7):
        indices = np.where(inputs == index)
        temp = np.zeros(shape=7, dtype=np.float32)
        temp[index] = 1
        target[indices[0].tolist(), indices[1].tolist(), indices[2].tolist(), :] = temp
    return target.transpose(0, 3, 1, 2)


def label_bluring(inputs):
    batchSize, numClass, height, width = inputs.shape
    outputs = np.ones((batchSize, numClass, height, width), dtype=np.float)
    for batchCnt in range(batchSize):
        for index in range(numClass):
            outputs[batchCnt, index, ...] = cv2.GaussianBlur(inputs[batchCnt, index, ...].astype(np.float), (7, 7), 0)
    return outputs

def inputImgTransBack(inputs):
    image = inputs[0].to("cpu")
    image[0] = image[0] + 0.3964
    image[1] = image[1] + 0.3695
    image[2] = image[2] + 0.2726
    #(image.numpy() * 255).transpose(1, 2, 0)
    return image


class MultiDataSet(data.Dataset):
    """input and label image dataset"""

    def __init__(self, root, cropSize, inSize, phase="train", labelExtension='.png', testFlag=False, final=False, preload=True):
        super(MultiDataSet, self).__init__()
        """
        Args:

        fileDir(string):  directory with all the input images.
        transform(callable, optional): Optional transform to be applied on a sample
        """
        self.root = root
        self.cropSize = cropSize
        self.inSize = inSize
        self.mean = np.array([0.3964, 0.3695, 0.2726])
        # self.fcn_mean = np.array([.485, .456, .406])
        # self.fcn_normal = np.array([.229, .224, .225])
        self.fileDir = join(self.root, phase)
        self.labelExtension = labelExtension
        self.testFlag = testFlag
        self.preload = preload
        self.final = final
        self.image_filenames = [image_name for image_name in listdir(self.fileDir + '/Sat') if
                                is_image_file(image_name)]
        if self.preload:
            self.images = []
            self.labels = []
            self._pre_load()
        self.classdict = {1: "urban", 2: "agriculture", 3: "rangeland", 4: "forest", 5: "water", 6: "barren",
                          0: "unknown"}

    def __getitem__(self, index):
        if not self.preload:
            timeStart = time.time()
            Satsample = cv2.imread(join(self.fileDir, "Sat/"+self.image_filenames[index]))
            image = cv2.cvtColor(Satsample, cv2.COLOR_BGR2RGB)

            #labelsample = cv2.imread(join(self.fileDir, 'Label/' + labelsamplename))
            #label = cv2.cvtColor(labelsample, cv2.COLOR_BGR2RGB)
            timeRead = time.time()
            #label = RGB_mapping_to_class(label)
            if not self.final:
                label = cv2.imread(join(self.fileDir, 'Notification/' +
                                        self.image_filenames[index].replace('_sat.jpg', '_mask.png')), 0).astype(np.int64)

            timeLabelTrans = time.time()
        else:
            image = self.images[index]
            label = self.labels[index]
        if not self.final:
            image, label = self._transform(image, label)
        else:
            image = self._transform(image)
        # imageLowReso = cv2.resize(
        #     image,
        #     (512, 512),
        #     interpolation=cv2.INTER_LINEAR,
        # )
        # imageLowReso = imageLowReso / 255 - self.mean
        # imageLowReso = imageLowReso.transpose(2, 0, 1)
        image = image / 255 - self.mean
        image = image.transpose(2, 0, 1)
        # image = image/255 - self.fcn_mean
        # for i, normal in enumerate(self.fcn_normal):
        #     image[i, ...] = image[i, ...] / normal
        timeFinish = time.time()
        # print("timeRead is %.2f \n timeLabelTrans is %.2f \n timeLabelTrans is %.2f \n" %
        #       (timeRead - timeStart, timeLabelTrans - timeRead, timeFinish - timeLabelTrans))
        if not self.final:
            return image.astype(np.float32), label.astype(np.int64)
        else:
            return image.astype(np.float32), self.image_filenames[index].replace("_sat.jpg", "_mask.png")

    def _transform(self, image, label=None):
        # Scaling
        # scale_factor = random.uniform(1, 2)
        # scale = math.ceil(scale_factor * self.cropSize)
        #
        if self.inSize != 2448:
            image = cv2.resize(
                image,
                (self.inSize, self.inSize),
                interpolation=cv2.INTER_LINEAR,
            )

        if not (self.testFlag or self.final):
            if self.inSize != 2448 and self.inSize != self.cropSize:
                    label = cv2.resize(
                        label,
                        (self.inSize, self.inSize),
                        interpolation=cv2.INTER_NEAREST,
                    )

            #scale
            if self.inSize == 2448 and self.cropSize < 2448:
                scaleSize = int(random.uniform(0.6, 1.4) * 2448)
            else:
                scaleSize = int(random.uniform(1, 1.2) * self.inSize)
            image = cv2.resize(
                image,
                (scaleSize, scaleSize),
                interpolation=cv2.INTER_NEAREST
            )
            label = cv2.resize(
                label,
                (scaleSize, scaleSize),
                interpolation=cv2.INTER_NEAREST,
            )

            h, w, _ = image.shape
            # Crop
            w_offset = random.randint(0, max(0, w - self.cropSize - 1))
            h_offset = random.randint(0, max(0, h - self.cropSize - 1))

            image = image[h_offset:h_offset + self.cropSize,
                          w_offset:w_offset + self.cropSize, :]
            label = label[h_offset:h_offset + self.cropSize,
                          w_offset:w_offset + self.cropSize]

            # Rotate
            rotate_time = np.random.randint(low=0, high=4)
            np.rot90(image, rotate_time)
            np.rot90(label, rotate_time)

            # Random flipping
            if random.random() < 0.5:
                image = np.fliplr(image).copy()  # HWC
                label = np.fliplr(label).copy()  # HW
        if not self.final:
            return image, label
        else:
            return image

    def _pre_load(self):
        print("preloading images and labels")
        for index, image_name in enumerate(self.image_filenames):
            print("loading image {}".format(index))
            Satsample = cv2.imread(join(self.fileDir, 'Sat/' + image_name))
            image = cv2.cvtColor(Satsample, cv2.COLOR_BGR2RGB)
            labelsamplename = find_label_map_name(image_name, self.labelExtension)
            labelsample = cv2.imread(join(self.fileDir, 'Label/' + labelsamplename))
            label = cv2.cvtColor(labelsample, cv2.COLOR_BGR2RGB)
            label = RGB_mapping_to_class(label)
            self.images.append(image.astype(np.float32))
            self.labels.append(label.astype(np.int64))
        print("finish preloading")

    def __len__(self):
        return len(self.image_filenames)

# dataset_train = MultiDataSet("/home/ckx9411sx/deepGlobe/land-train")
# np.save('/home/ckx9411sx/deepGlobe/temp/train_data_mc.npy', dataset_train)
# print(dataset_train)
