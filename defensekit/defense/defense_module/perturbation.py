import numpy as np
import os
import random
import string
from PIL import ImageDraw
import torchvision.transforms as T

class Perturbation:
    def __init__(self, pert_type="MaskImage", ):
        super().__init__()
        self.perturbation_fn = globals()[pert_type]()

    def __call__(self, instances):
        self.perturbation_fn(instances)


class ImagePerturbation:
    """Base class for random perturbations on image."""

    def __init__(self):
        pass

    def sample_odd_level(self, maxl,minl=1):
        return 2*np.random.randint(low=minl, high=maxl)-1
    
    def sample_float_level(self, maxl, minl=0.1):
        return np.random.uniform(low=minl, high=maxl)
    
    def sample_int_level(self, maxl, minl=1):
        return np.random.randint(low=minl, high=maxl)
    
    def __call__(self, instances):
        for instance in instances:
            instance.image = self.perturb_img(instance.image)


class MaskImage(ImagePerturbation):
    def __init__(self):
        super(MaskImage, self).__init__()

    def load_position(self, position,size,mask_size):
        if position=='rand':
            # pos1=np.random.randint(0,int(size[0]/4))
            # pos2=np.random.randint(int(3*size[1]/4),int(size[1])) # down left
            pos1=np.random.randint(0,int(size[0]-mask_size[0]))
            pos2=np.random.randint(0,int(size[1]-mask_size[1])) # down left
            return (pos1,pos2)
        else:
            size_tuple=(min(int(position.split('-')[0]),size[0]),min(int(position.split('-')[1]),size[1]))
            return size_tuple
        
    def generate_mask_pil(self, image,mask_type,mask_size,position):
        # Create a blank mask
        # mask = Image.new("1", image.size, 0)
        # draw = ImageDraw.Draw(mask)
        draw = ImageDraw.Draw(image)
        if mask_type=='r':
            # Define the rectangle coordinates and dimensions
            rect_x, rect_y, rect_width, rect_height = position[0], position[1], mask_size[0], mask_size[1]
            # Draw a rectangle on the mask
            draw.rectangle((rect_x, rect_y, rect_x + rect_width, rect_y + rect_height), fill='black') # fill = 1
            output_image=image
        else:
            print('not support mask!!')
            os._exit(0)
        # if mask.mode == 'RGBA':
        #     mask = mask.split()[-1]  # Use the alpha channel as the mask
        # output_image = Image.new("RGB", image.size)
        # output_image.paste(image, (0, 0), mask)
        return output_image

    def perturb_img(self, img,position='rand',mask_type='r',mask_size=(200,200)):
        img_size=img.size # width, height
        new_mask_size=[min(mask_size[sz],(0.3*img_size[sz])) for sz in range(len(mask_size))]
        # mask size should not larget than 0.5*image size
        position=self.load_position(position,img_size,new_mask_size)
        # generate mask
        new_image = self.generate_mask_pil(img,mask_type,new_mask_size,position)
        # new_image = generate_dark_mask(img,mask_type,mask_size,position)
        return new_image


class BlurImage(ImagePerturbation):
    def __init__(self):
        super(BlurImage, self).__init__()

    def perturb_img(self, img, level=5):
        # level: int: 5 --> kernel size
        k1=self.sample_odd_level(level)
        k2=self.sample_odd_level(level)
        transform = T.GaussianBlur(kernel_size=(k1, k2))
        new_image = transform(img)
        return new_image
    

class FlipImage(ImagePerturbation):
    def __init__(self):
        super(FlipImage, self).__init__()

    def perturb_img(self, img, level=1.0):
        # level: float: 1.0 --> p
        p=self.sample_float_level(level)
        transform = T.RandomHorizontalFlip(p=p)
        new_image = transform(img)
        return new_image
    

class VFlipImage(ImagePerturbation):
    def __init__(self):
        super(VFlipImage, self).__init__()

    def perturb_img(self, img, level=1.0):
        # level: float: 1.0 --> p
        p = self.sample_float_level(level)
        transform = T.RandomVerticalFlip(p=p)
        new_image = transform(img)
        return new_image


class ResizeCropImage(ImagePerturbation):
    def __init__(self):
        super(ResizeCropImage, self).__init__()

    def perturb_img(self, img, level=500):
        # level: int: 500 --> image size
        size = img.size
        s1 = int(max(self.sample_int_level(level), 0.8 * size[0]))
        s2 = int(max(self.sample_int_level(level), 0.8 * size[1]))
        transform = T.RandomResizedCrop((s2, s1), scale=(0.9, 1))
        new_image = transform(img)
        return new_image


class GrayImage(ImagePerturbation):
    def __init__(self):
        super(GrayImage, self).__init__()

    def perturb_img(self, img, level=1):
        rate = self.sample_float_level(level)
        if rate >= 0.5:
            transform = T.Grayscale(num_output_channels=len(img.split()))
            new_image = transform(img)
        else:
            new_image = img
        return new_image


class RotationImage(ImagePerturbation):
    def __init__(self):
        super(RotationImage, self).__init__()

    def perturb_img(self, img, level=180):
        rate = self.sample_float_level(level)
        transform = T.RandomRotation(degrees=(0, rate))
        new_image = transform(img)
        return new_image


class ColorJitterImage(ImagePerturbation):
    def __init__(self):
        super(ColorJitterImage, self).__init__()

    def perturb_img(self, img, level1=1, level2=0.5):
        rate1 = self.sample_float_level(level1)
        rate2 = self.sample_float_level(level2)
        transform = T.ColorJitter(brightness=rate1, hue=rate2)
        new_image = transform(img)
        return new_image


class SolarizeImage(ImagePerturbation):
    def __init__(self):
        super(SolarizeImage, self).__init__()

    def perturb_img(self, img, level=200):
        rate = self.sample_float_level(level)
        transform = T.RandomSolarize(threshold=rate)
        new_image = transform(img)
        return new_image


class PosterizeImage(ImagePerturbation):
    def __init__(self):
        super(PosterizeImage, self).__init__()

    def perturb_img(self, img, level=3):
        rate = self.sample_int_level(level)
        transform = T.RandomPosterize(bits=rate)
        new_image = transform(img)
        return new_image


class PolicyAugImage(ImagePerturbation):
    def __init__(self, img_aug_dict):
        super(PolicyAugImage, self).__init__()
        self.img_aug_dict = img_aug_dict

    def find_index(self, L, b):
        for i in range(len(L) - 1):
            if b > L[i] and b < L[i + 1]:
                return i
        return -1

    def perturb_img(self, img, level='0.34-0.45-0.21', pool='RR-BL-RP'):
        mutator_list = [self.img_aug_dict[_mut] for _mut in pool.split('-')]
        probability_list = [float(_value) for _value in level.split('-')]
        probability_list = [sum(probability_list[:i+1]) for i in range(len(probability_list))]
        randnum = np.random.random()
        index = self.find_index(probability_list, randnum)
        
        new_image = mutator_list[index](img)
        return new_image


class WordsPerturbation:

    """Base class for random perturbations."""

    def __init__(self, q=10):
        self.q = q
        self.alphabet = string.printable

    def __call__(self, instances):
        for instance in instances:
            instance.question = self.perturb_words(instance.question)

class RandomSwapPerturbation(WordsPerturbation):

    """Implementation of random swap perturbations.
    See `RandomSwapPerturbation` in lines 1-5 of Algorithm 2."""

    def __init__(self, q=10):
        super(RandomSwapPerturbation, self).__init__(q)

    def perturb_words(self, s):
        list_s = list(s)
        sampled_indices = random.sample(range(len(s)), int(len(s) * self.q / 100))
        for i in sampled_indices:
            list_s[i] = random.choice(self.alphabet)
        return ''.join(list_s)

class RandomPatchPerturbation(WordsPerturbation):
    """Implementation of random patch perturbations.
    See `RandomPatchPerturbation` in lines 6-10 of Algorithm 2."""

    def __init__(self, q=10):
        super(RandomPatchPerturbation, self).__init__(q)

    def perturb_words(self, s):
        list_s = list(s)
        substring_width = int(len(s) * self.q / 100)
        max_start = len(s) - substring_width
        start_index = random.randint(0, max_start)
        sampled_chars = ''.join([
            random.choice(self.alphabet) for _ in range(substring_width)
        ])
        list_s[start_index:start_index+substring_width] = sampled_chars
        return ''.join(list_s)

class RandomInsertPerturbation(WordsPerturbation):
    """Implementation of random insert perturbations.
    See `RandomPatchPerturbation` in lines 11-17 of Algorithm 2."""

    def __init__(self, q=10):
        super(RandomInsertPerturbation, self).__init__(q)

    def perturb_words(self, s):
        list_s = list(s)
        sampled_indices = random.sample(range(len(s)), int(len(s) * self.q / 100))
        for i in sampled_indices:
            list_s.insert(i, random.choice(self.alphabet))
        return ''.join(list_s)