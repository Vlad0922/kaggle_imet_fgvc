import math
import imgaug
from PIL import Image
import numpy as np

class RandomGaussianNoise(object):
    def __init__(self, p=0.5, sigma=1):
        self.sigma = sigma
        self.p = 0.5
        
    def __call__(self, img):
        prob = np.random.uniform()
        
        if(prob > self.p):
            return img
        
        row, col = img.size
        noise = np.random.normal(0.0, self.sigma, (col,row,3))

        img_trans = np.clip(img + noise, 0, 255)
        return Image.fromarray(img_trans.astype('uint8'))
    
class RandomGaussianBlur(object):
    def __init__(self, p=0.5, sigma=1):
        self.p = p
        self.sigma = sigma
        self.augmenter = imgaug.augmenters.GaussianBlur(sigma)
    
    def __call__(self, img):
        prob = np.random.uniform()
        
        if(prob > self.p):
            return img
        
        img_trans = self.augmenter.augment_image(np.array(img))
        return Image.fromarray(img_trans.astype('uint8'))
    
class RandomRotation(object):
    def __init__(self, p=0.5, degrees=30):
        self.p = p
        self.degrees = degrees
        self.augmenter = transforms.RandomRotation(degrees)
        
    def __call__(self, img):
        prob = np.random.uniform()
        
        if prob > self.p:
            return img
        
        return self.augmenter(img)
    
class RandomAffine(object):
    def __init__(self, p=0.5, degrees=30):
        self.p = p
        self.degrees = degrees
        self.augmenter = transforms.RandomAffine(degrees=degrees)
    
    def __call__(self, img):
        prob = np.random.uniform()
        
        if prob > self.p:
            return img
        
        return self.augmenter(img) 

class RandomHueSaturation(object):
    def __init__(self, p=0.5, min=-10, max=10):
        self.p = p
        self.min = min
        self.max = max
    
    def __call__(self, img):
        prob = np.random.uniform()
        
        if prob > self.p:
            return img
        
        augmenter = imgaug.augmenters.AddToHueAndSaturation(int(np.random.uniform(self.min, self.max)))
        
        img_trans = augmenter.augment_image(np.array(img))
        return Image.fromarray(img_trans.astype('uint8'))
    
class RandomCoarseDropout(object):
    def __init__(self, p=0.5, drop_prob=0.1, size_percent=0.2):
        self.p = p
        self.drop_prob = drop_prob
        self.augmenter = imgaug.augmenters.CoarseDropout(drop_prob, size_percent=size_percent)
    
    def __call__(self, img):
        prob = np.random.uniform()
        
        if(prob > self.p):
            return img
        
        img_trans = self.augmenter.augment_image(np.array(img))
        return Image.fromarray(img_trans.astype('uint8'))

    
class RandomBrightness(object):
    def __init__(self, p=0.5, min=-50, max=50):
        self.p = p
        self.min = min
        self.max = max
    
    def __call__(self, img):
        prob = np.random.uniform()
        
        if(prob > self.p):
            return img
        
        augmenter = imgaug.augmenters.Add(int(np.random.uniform(self.min, self.max)))
        
        img_trans = augmenter.augment_image(np.array(img))
        return Image.fromarray(img_trans.astype('uint8'))
    
class RandomGamma(object):
    def __init__(self, p=0.5, min=0.5, max=1.5):
        self.p = p
        self.min = min
        self.max = max
    
    def __call__(self, img):
        prob = np.random.uniform()
        
        if(prob > self.p):
            return img
        
        augmenter = imgaug.augmenters.GammaContrast(np.random.uniform(self.min, self.max))
        
        img_trans = augmenter.augment_image(np.array(img))
        return Image.fromarray(img_trans.astype('uint8'))
    
class RandomSizedCrop:
    """Random crop the given PIL.Image to a random size
    of the original size and and a random aspect ratio
    of the original aspect ratio.
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR,
                 min_aspect=4/5, max_aspect=5/4,
                 min_area=0.25, max_area=1):
        self.size = size
        self.interpolation = interpolation
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.min_area = min_area
        self.max_area = max_area

    def __call__(self, img):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(self.min_area, self.max_area) * area
            aspect_ratio = random.uniform(self.min_aspect, self.max_aspect)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))

                return img.resize((self.size, self.size), self.interpolation)

        # Fallback
        scale = transforms.Resize(self.size, interpolation=self.interpolation)
        crop = transforms.CenterCrop(self.size)
        return crop(scale(img))