
# coding: utf-8

# In[1]:



# coding: utf-8

# In[ ]:


from PIL import Image
from os import listdir, walk
from os.path import isfile, join
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets

# add root path of dataset
# structure should be: DATA_PATH="/some/path"
# /some/path/class0/*.jpg
# /some/path/class1/*.jpg
# ... 
DATA_PATH = "Training"


# run once
def get_file_paths(dir_name):
    for (dirpath, dirnames, filenames) in walk(dir_name):
        for filename in filenames:
            ret = join(dirpath, filename)
            if isfile(ret):
                yield ret


def get_images_ndarray(dir_name):
    for path in get_file_paths(dir_name):
        if path.endswith(".jpg") or path.endswith(".jpeg"):
            img = Image.open(path)
            yield np.array(img)   # [height, width, channel]


def compute_mean_std():
    num_of_img = 0
    sums = 0
    nums = 0
    for image in get_images_ndarray(DATA_PATH):
        num_of_img += 1
        print(num_of_img)
        val = image.flatten()
        sums += np.sum(val)
        nums += len(val)
    mean = sums / nums
    sums = 0
    for image in get_images_ndarray(DATA_PATH):
        val = image.flatten()
        sums += np.sum(np.square(val - mean))
    std = np.sqrt(sums / nums)
    
    print(num_of_img)
    print("Mean for channels 0, 1, 2: %s"%str(mean))
    print("Std for channels 0, 1, 2: %s"%str(std))
    return mean, std

mean, std = compute_mean_std()

'''
normalize = transforms.Normalize(mean=[10, 10, 10, std=std)

train_loader = DataLoader(
    datasets.ImageFolder(DATA_PATH, transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=False)

for data in train_loader:
    print(data)
'''


