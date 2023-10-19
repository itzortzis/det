import numpy as np
from skimage import exposure
from skimage.filters import difference_of_gaussians


class Det():
    
    def __init__(self, dataset):
        self.dataset = dataset
        self.channels = 4
        
    
    def __len__(self):
        return self.dataset.shape[0]
    
    
    def gamma(self, img):
        return exposure.adjust_gamma(img, 2)
    
    
    def histo(self, img):
        return exposure.equalize_hist(img)
    
    
    def gaussian_diff(self, img):
        return difference_of_gaussians(img, 10, 15)
    
    
    def enhance_image(self, img):
        h = img.shape[0]
        w = img.shape[1]
        e_img = np.zeros((h, w, self.channels))
        e_img[:, :, 0] = img
        e_img[:, :, 1] = self.gamma(img)
        e_img[:, :, 2] = self.histo(img)
        e_img[:, :, 3] = self.gaussian_diff(img)
        
        return e_img
    
    
    def enhance_dataset(self):
        h = self.dataset.shape[1]
        w = self.dataset.shape[2]
        e_dataset = np.zeros((self.__len__(), h, w, self.channels + 1))
        for i in range(len(self.dataset)):
            e_dataset[i, :, :, :-1] = self.enhance_image(self.dataset[i, :, :, 0])
            e_dataset[i, :, :, -1] = self.dataset[i, :, :, 1]
        
        return e_dataset