from torchvision import datasets
import numpy as np

class DGFolder(datasets.ImageFolder):

    def __init__(self, root, transform):
        super(DGFolder, self).__init__(root, transform)
        targets = np.asarray([s[1] for s in self.samples])
        self.targets = targets
        self.img_num = len(self.samples)
        print(self.img_num)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample1 = self.loader(path)
        sample2 = sample1
        if self.transform is not None:
            sample1 = self.transform(sample1)
            sample2 = self.transform(sample2)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample1, sample2, target


