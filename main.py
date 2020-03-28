from SimpleDataset import SimpleDataset 
from myOwnDataset import myOwnDataset 


# data directory
root = "my_data"

# assume we have 3 jpg images
filenames = ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg']

# the class of image might be ['black cat', 'tabby cat', 'tabby cat']
labels = [0, 1, 2, 0]

# create own Dataset
my_dataset = SimpleDataset(root=root,
                           filenames=filenames,
                           labels=labels
                           )


import os
import torch
from torch.utils import data
from PIL import Image
from torchvision import transforms

batch_size = 1
num_workers = 4

data_loader = torch.utils.data.DataLoader(my_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=num_workers
                                         )


import numpy as np
import matplotlib.pyplot as plt

for images, labels in data_loader:
    # image shape is [batch_size, 3 (due to RGB), height, width]
    img = transforms.ToPILImage()(images[0])
    plt.imshow(img)
    plt.show()
    print(labels)


