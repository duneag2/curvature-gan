import torch.utils.data as data

from PIL import Image

import os
import os.path
import random

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, extensions):
    
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if has_file_allowed_extension(fname, extensions):
                    fi = fname.split("_")[0]                   
                    #if fi == fib:
                    #print(fi)
                    path = os.path.join(root, fname)
                    item = (path, 0)
                    images.append(item)

    return images


def make_datasetB(dir, extensions):
    
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if has_file_allowed_extension(fname, extensions):
                    fi = fname.split("_")[0]                   
                    #if fi == fib:
                    #print(fi)
                    path = os.path.join(root, fname)
                    item = (path, 0)
                    images.append(item)

    return images

class DatasetFolder(data.Dataset):
    def __init__(self, root, loader, extensions, transform=None, target_transform=None):
        # classes, class_to_idx = find_classes(root)
#         rootA = root.split("_")[0]
#         print(root)
#         print(rootA)
        rootA=root

        samples = make_dataset(rootA, extensions)
        samplesB = make_datasetB(rootB, extensions)
        
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + rootA + "\n"
                               "Supported extensions are: " + ",".join(extensions)))
        if len(samplesB) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + rootB + "\n"
                               "Supported extensions are: " + ",".join(extensions)))
        self.opt = root
        self.root = rootA
        self.loader = loader
        self.extensions = extensions
        self.samples = samples
        self.samplesB = samplesB

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        fiopt = self.opt.split("/")[-1]

        #print(fiopt,index)

        patha, targeta = self.samples[index]
#         print(self.samples[index])
        
        pathbarr = []
        fia = patha.split("/")[-1].split("_")
        for i in range(0,len(self.samplesB)):
            im, n = self.samplesB[i]
            fib = im.split("/")[-1].split("_")
            if (fia[0] == fib[0]) & (fia[-1] == fib[-1]): 
                pathb, targetb = self.samplesB[i]
                pathbarr.append(pathb)
        try:
            num = random.randint(0,len(pathbarr)-1)
            pathb = pathbarr[num]
        except:
            pathb, targetb = self.samplesB[index]
         
        print(patha, targeta,"________________________________")
        print(pathb, targetb,"________________________________")

        samplea = self.loader(patha)
        sampleb = self.loader(pathb)

        if self.transform is not None:
            samplea = self.transform(samplea)
            sampleb = self.transform(sampleb)
        if self.target_transform is not None:
            targeta = self.target_transform(targeta)
            targetb = self.target_transform(targetb)

        return samplea, sampleb, targeta, targetb 

    def __len__(self):
        return len(self.samplesB)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        print(fmt_str)
        return fmt_str

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def default_loader(path):
    return pil_loader(path)

class ImageFolder(DatasetFolder):
    def __init__(self, root, transform=None, target_transform=None, 
                 loader=default_loader):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform
                                          )
        self.imgs = self.samples
        self.imgsB = self.samplesB
