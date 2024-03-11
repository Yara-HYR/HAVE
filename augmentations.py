import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as TF
import torchvision.datasets as datasets
import kornia
import utils
import os

import random
import torch.fft
from skimage.util.shape import view_as_windows
places_dataloader = None
places_iter = None
import torch.nn as nn
from TransformLayer import ColorJitterLayer
import json

def load_config(key=None):
	path = os.path.join('setup', 'config.cfg')
	with open(path) as f:
		data = json.load(f)
	if key is not None:
		return data[key]
	return data

	


def _load_places(batch_size=256, image_size=84, num_workers=16, use_val=False):
	global places_dataloader, places_iter
	partition = 'val' if use_val else 'train'
	print(f'Loading {partition} partition of places365_standard...')
	for data_dir in load_config('datasets'):
		if os.path.exists(data_dir):
			fp = os.path.join(data_dir, 'places365_standard', partition)
			if not os.path.exists(fp):
				print(f'Warning: path {fp} does not exist, falling back to {data_dir}')
				fp = data_dir
			places_dataloader = torch.utils.data.DataLoader(
				datasets.ImageFolder(fp, TF.Compose([
					TF.RandomResizedCrop(image_size),
					TF.RandomHorizontalFlip(),
					TF.ToTensor()
				])),
				batch_size=batch_size, shuffle=True,
				num_workers=num_workers, pin_memory=True)
			places_iter = iter(places_dataloader)
			break
	if places_iter is None:
		raise FileNotFoundError('failed to find places365 data at any of the specified paths')
	print('Loaded dataset from', data_dir)


def _get_places_batch(batch_size):
	global places_iter
	try:
		imgs, _ = next(places_iter)
		if imgs.size(0) < batch_size:
			places_iter = iter(places_dataloader)
			imgs, _ = next(places_iter)
	except StopIteration:
		places_iter = iter(places_dataloader)
		imgs, _ = next(places_iter)
	return imgs.cuda()


def random_overlay(x, dataset='places365_standard'):
	"""Randomly overlay an image from Places"""
	global places_iter
	alpha = 0.1

	if dataset == 'places365_standard':
		if places_dataloader is None:
			_load_places(batch_size=x.size(0), image_size=[x.size(-2),x.size(-1)])
		imgs = _get_places_batch(batch_size=x.size(0)).repeat(1, x.size(1)//3, 1, 1)
	else:
		raise NotImplementedError(f'overlay has not been implemented for dataset "{dataset}"')

	return ((1-alpha)*(x/255.) + (alpha)*imgs)*255.


def random_conv(x):
	"""Applies a random conv2d, deviates slightly from https://arxiv.org/abs/1910.05396"""
	n, c, h, w = x.shape
	for i in range(n):
		weights = torch.randn(3, 3, 3, 3).to(x.device)
		temp_x = x[i:i+1].reshape(-1, 3, h, w)/255.
		temp_x = F.pad(temp_x, pad=[1]*4, mode='replicate')
		out = torch.sigmoid(F.conv2d(temp_x, weights))*255.
		total_out = out if i == 0 else torch.cat([total_out, out], axis=0)
	return total_out.reshape(n, c, h, w)

def random_convolution(imgs):
    '''
    random covolution in "network randomization"
    
    (imbs): B x (C x stack) x H x W, note: imgs should be normalized and torch tensor
    '''

    # p = random.uniform(0, 1)

    # if p > 0.5:
    #     return imgs    
    _device = imgs.device
    
    img_h, img_w = imgs.shape[2], imgs.shape[3]
    num_stack_channel = imgs.shape[1]
    num_batch = imgs.shape[0]
    num_trans = num_batch
    batch_size = int(num_batch / num_trans)
    
    # initialize random covolution
    rand_conv = nn.Conv2d(3, 3, kernel_size=3, bias=False, padding=1).to(_device)
    
    for trans_index in range(num_trans):
        torch.nn.init.xavier_normal_(rand_conv.weight.data)
        temp_imgs = imgs[trans_index*batch_size:(trans_index+1)*batch_size]
        temp_imgs = temp_imgs.reshape(-1, 3, img_h, img_w) # (batch x stack, channel, h, w)
        rand_out = rand_conv(temp_imgs)
        if trans_index == 0:
            total_out = rand_out
        else:
            total_out = torch.cat((total_out, rand_out), 0)
    total_out = total_out.reshape(-1, num_stack_channel, img_h, img_w)
    return total_out
def batch_from_obs(obs, batch_size=32):
	"""Copy a single observation along the batch dimension"""
	if isinstance(obs, torch.Tensor):
		if len(obs.shape)==3:
			obs = obs.unsqueeze(0)
		return obs.repeat(batch_size, 1, 1, 1)

	if len(obs.shape)==3:
		obs = np.expand_dims(obs, axis=0)
	return np.repeat(obs, repeats=batch_size, axis=0)


def prepare_pad_batch(obs, next_obs, action, batch_size=32):
	"""Prepare batch for self-supervised policy adaptation at test-time"""
	batch_obs = batch_from_obs(torch.from_numpy(obs).cuda(), batch_size)
	batch_next_obs = batch_from_obs(torch.from_numpy(next_obs).cuda(), batch_size)
	batch_action = torch.from_numpy(action).cuda().unsqueeze(0).repeat(batch_size, 1)

	return random_crop_cuda(batch_obs), random_crop_cuda(batch_next_obs), batch_action


def identity(x):
	return x


def random_shift(imgs, pad=4):
	"""Vectorized random shift, imgs: (B,C,H,W), pad: #pixels"""
	_,_,h,w = imgs.shape
	imgs = F.pad(imgs, (pad, pad, pad, pad), mode='replicate')
	return kornia.augmentation.RandomCrop((h, w))(imgs)


def mix_data(x, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    
    p = random.uniform(0, 1)

    if p > 0.5:
        return x

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    
    x0,x1,x2 = torch.chunk(x,3,dim=1)
    x = torch.cat((x0.unsqueeze(1),x1.unsqueeze(1),x2.unsqueeze(1)),dim=1)
    fft_1 = torch.fft.fftn(x, dim=(2,3,4))


    # print(len(fft_1[0]))
    abs_1, angle_1 = torch.abs(fft_1), torch.angle(fft_1)

    fft_2 = torch.fft.fftn(x[index, :], dim=(2,3,4))
    abs_2, angle_2 = torch.abs(fft_2), torch.angle(fft_2)

    fft_1 = abs_2*torch.exp((1j) * angle_1)

    mixed_x = torch.fft.ifftn(fft_1, dim=(2,3,4)).float()
    x0,x1,x2 = torch.chunk(mixed_x,3,dim=1)
    x = torch.cat((x0.squeeze(1),x1.squeeze(1),x2.squeeze(1)),dim=1)
    return x

def get_phase(x, use_cuda=True):
#     '''Returns mixed inputs, pairs of targets, and lambda'''
    
    # p = random.uniform(0, 1)

    # if p > 0.5:
    #     return x

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    # fft_1 = torch.fft.fftn(x, dim=(1,2,3))
    # abs_1, angle_1 = torch.abs(fft_1), torch.angle(fft_1)

    # # fft_2 = torch.fft.fftn(x[index, :], dim=(1,2,3))
    # # abs_2, angle_2 = torch.abs(fft_2), torch.angle(fft_2)

    # # fft_1 = abs_2*torch.exp((1j) * angle_1)

    # # mixed_x = torch.fft.ifftn(fft_1, dim=(1,2,3)).float()

    # return angle_1

    x0,x1,x2 = torch.chunk(x,3,dim=1)
    x = torch.cat((x0.unsqueeze(1),x1.unsqueeze(1),x2.unsqueeze(1)),dim=1)
    fft_1 = torch.fft.fftn(x, dim=(2,3,4))
    abs_1, angle_1 = torch.abs(fft_1), torch.angle(fft_1)

    # fft_2 = torch.fft.fftn(x[index, :], dim=(2,3,4))
    # abs_2, angle_2 = torch.abs(fft_2), torch.angle(fft_2)

    # fft_1 = abs_2*torch.exp((1j) * angle_1)

    # mixed_x = torch.fft.ifftn(fft_1, dim=(2,3,4)).float()
    # x0,x1,x2 = torch.chunk(mixed_x,3,dim=1)
    # x = torch.cat((x0.squeeze(1),x1.squeeze(1),x2.squeeze(1)),dim=1)
    return angle_1


# def random_color(imgs,pad=4,brightness=0.0,contrast=0.0,saturation=0.0,hue=0.0,color_p=0.1,gray_p=0.1,solar_p=0.1,poster_p=0.1,sharp_p=0.1,gray_equa=0.1):
# 	brightness=random.choices(population=[3,6,7,9], weights=None, k=1)[0]
# 	contrast=random.choices(population=[1,2,3,4,5,6], weights=None, k=1)[0]
# 	saturation=random.choices(population=[0,1,2,3], weights=None, k=1)[0]
# 	hue= random.choices(population=[-0.2,-0.1, 0.1,0.2], weights=None, k=1)[0]
# 	_,_,h,w = imgs.shape
# 	imgs = F.pad(imgs, (pad, pad, pad, pad), mode='replicate')
# 	imgs = kornia.augmentation.RandomCrop((h, w))(imgs)
# 	imgs = kornia.augmentation.ColorJitter(brightness,contrast,saturation,hue,p=color_p)(imgs)
# 	# Change the brightness, contrast, saturation and hue randomly given tensor image or a batch of tensor images.
# 	# Input should be a tensor of shape (C, H, W) or a batch of tensors (B,C,H,W).
# 	imgs = kornia.augmentation.RandomGrayscale(p=gray_p)(imgs)#Random Grayscale transformation according to a probability p value
# 	imgs = kornia.augmentation.RandomSolarize(p=solar_p)(imgs)#
# 	imgs = kornia.augmentation.RandomPosterize(p = poster_p)(imgs)
# 	imgs = kornia.augmentation.RandomSharpness(p=sharp_p)(imgs)
# 	imgs = kornia.augmentation.RandomEqualize(p = gray_equa)(imgs)
# 	#RandomMotionBlur
# 	#RandomGaussianBlur                                     │
# 	return imgs


def random_color(imgs,pad=4,brightness=0.0,contrast=0.0,saturation=0.0,hue=0.0,color_p=0.1,gray_p=0.2,solar_p=0.2,poster_p=0.2,sharp_p=0.2,equa_p=0.1):
	b,c,h,w = imgs.shape
#	imgs = F.pad(imgs, (pad, pad, pad, pad), mode='replicate')
#	imgs = kornia.augmentation.RandomCrop((h, w))(imgs)

	# aug_k=random.choices(population=[0,1,2,3,4,5,6], weights=None, k=1)[0]



	aug_k=random.choices(population=[0,1,2,3,4,5], weights=None, k=1)[0]
	if aug_k==0 or aug_k==1 or aug_k==2:
		aug=random.choices(population=[kornia.augmentation.RandomSharpness(p=0.5),kornia.augmentation.RandomGaussianNoise(p=0.5),  kornia.augmentation.RandomGaussianBlur((3, 3), (0.1, 2.0), p=0.5)], weights=None, k=1)[0]
		return aug(imgs)

	elif aug_k==3:
		aug=random.choices(population=[kornia.augmentation.RandomSharpness(p=0.5),kornia.augmentation.RandomGaussianNoise(p=0.5),  kornia.augmentation.RandomGaussianBlur((3, 3), (0.1, 2.0), p=0.5),\
			      kornia.augmentation.RandomErasing((0.018, 0.018), (.1, 1/.8), p=0.2), random_grayscale, random_convolution, random_color_jitter,random_overlay], weights=None, k=1)[0]
		return aug(imgs)
	else:
		return imgs
	# imgs = aug(imgs)

	# aug =  nn.Sequential(
	# 	# kornia.augmentation.RandomGrayscale(p=gray_p),
	# 	# kornia.augmentation.ColorJitter(0.2, 0.3, 0.2, 0.3),
	# 	kornia.augmentation.RandomSharpness(p=sharp_p),
	# 	# kornia.augmentation.RandomEqualize(p=equa_p),
	# 	kornia.augmentation.RandomGaussianBlur((3, 3), (0.1, 2.0), p=0.2),
	# 	#kornia.augmentation.RandomErasing((.1, .1), (.3, 1/.3), p=0.5),
	# 	kornia.augmentation.RandomErasing((0.018, 0.018), (.1, 1/.8), p=0.2),
	# 	# kornia.augmentation.RandomErasing((0.018, 0.018), (.1, 1/.8), p=0.2),
	# 	# kornia.augmentation.RandomErasing((0.018, 0.018), (.1, 1/.8), p=0.2),
	# 	# kornia.augmentation.RandomErasing((0.018, 0.018), (.1, 1/.8), p=0.2),	
	# 	kornia.augmentation.RandomGaussianNoise(p=0.2)	
	# 	)

	# import pdb
	# pdb.set_trace()
	         
	# if aug_k==0:
	# 	aug=kornia.augmentation.RandomSharpness(p=0.5)
	# 	imgs = aug(imgs)
	# elif aug_k==1:
	# 	aug = kornia.augmentation.RandomGaussianNoise(p=0.5)	
	# 	imgs = aug(imgs)
	# elif aug_k==2:
	# 	aug = kornia.augmentation.RandomGaussianBlur((3, 3), (0.1, 2.0), p=0.5)
	# 	imgs = aug(imgs)
	# elif aug_k ==3:
	# 	aug=random_convolution
	# 	imgs = aug(imgs)
	# elif aug_k==4:
	# 	aug = kornia.augmentation.RandomGrayscale(p=0.5)  
	# 	imgs = aug(imgs)
	# elif aug_k==5:
	# 	aug =  random_color_jitter#kornia.augmentation.ColorJitter(0.4,0.4,0.4,0.5)
	# 	imgs = aug(imgs)
	# elif aug_k==6:
	# 	kornia.augmentation.RandomErasing((0.018, 0.018), (.1, 1/.8), p=0.2)
	# 	imgs = aug(imgs)
	# # elif aug_k==
	# # print(aug)        



	# imgs = aug(imgs)
	# return imgs

# def random_crop(x, size=84, w1=None, h1=None, return_w1_h1=False):
# 	"""Vectorized CUDA implementation of random crop, imgs: (B,C,H,W), size: output size"""
# 	assert (w1 is None and h1 is None) or (w1 is not None and h1 is not None), \
# 		'must either specify both w1 and h1 or neither of them'
# 	assert isinstance(x, torch.Tensor) and x.is_cuda, \
# 		'input must be CUDA tensor'
	
# 	n = x.shape[0]
# 	img_size = x.shape[-1]
# 	crop_max = img_size - size

# 	if crop_max <= 0:
# 		if return_w1_h1:
# 			return x, None, None
# 		return x

# 	x = x.permute(0, 2, 3, 1)

# 	if w1 is None:
# 		w1 = torch.LongTensor(n).random_(0, crop_max)
# 		h1 = torch.LongTensor(n).random_(0, crop_max)

# 	windows = view_as_windows_cuda(x, (1, size, size, 1))[..., 0,:,:, 0]
# 	cropped = windows[torch.arange(n), w1, h1]

# 	if return_w1_h1:
# 		return cropped, w1, h1

# 	return cropped




	
def random_crop(imgs, out=84):
    """
        args:
        imgs: np.array shape (B,C,H,W)
        out: output size (e.g. 84)
        returns np.array
    """
    n, c, h, w = imgs.shape
    crop_max = h - out + 1
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    cropped = np.empty((n, c, out, out), dtype=imgs.dtype)
    for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
        
        cropped[i] = img[:, h11:h11 + out, w11:w11 + out]
    return cropped

def random_translate(imgs, size, return_random_idxs=False, h1s=None, w1s=None):
    n, c, h, w = imgs.shape
    assert size >= h and size >= w
    outs = np.zeros((n, c, size, size), dtype=imgs.dtype)
    h1s = np.random.randint(0, size - h + 1, n) if h1s is None else h1s
    w1s = np.random.randint(0, size - w + 1, n) if w1s is None else w1s
    for out, img, h1, w1 in zip(outs, imgs, h1s, w1s):
        out[:, h1:h1 + h, w1:w1 + w] = img
    if return_random_idxs:  # So can do the same to another set of imgs.
        return outs, dict(h1s=h1s, w1s=w1s)
    return outs
def grayscale(imgs):
    # imgs: b x c x h x w
    device = imgs.device
    b, c, h, w = imgs.shape
    frames = c // 3
    
    imgs = imgs.view([b,frames,3,h,w])
    imgs = imgs[:, :, 0, ...] * 0.2989 + imgs[:, :, 1, ...] * 0.587 + imgs[:, :, 2, ...] * 0.114 
    
    imgs = imgs.type(torch.uint8).float()
    # assert len(imgs.shape) == 3, imgs.shape
    imgs = imgs[:, :, None, :, :]
    imgs = imgs * torch.ones([1, 1, 3, 1, 1], dtype=imgs.dtype).float().to(device) # broadcast tiling
    return imgs

def random_grayscale(images,p=.3):
    """
        args:
        imgs: torch.tensor shape (B,C,H,W)
        device: cpu or cuda
        returns torch.tensor
    """
    device = images.device
    in_type = images.type()
    images = images * 255.
    images = images.type(torch.uint8)
    # images: [B, C, H, W]
    bs, channels, h, w = images.shape
    images = images.to(device)
    gray_images = grayscale(images)
    rnd = np.random.uniform(0., 1., size=(images.shape[0],))
    mask = rnd <= p
    mask = torch.from_numpy(mask)
    frames = images.shape[1] // 3
    images = images.view(*gray_images.shape)
    mask = mask[:, None] * torch.ones([1, frames]).type(mask.dtype)
    mask = mask.type(images.dtype).to(device)
    mask = mask[:, :, None, None, None]
    out = mask * gray_images + (1 - mask) * images
    out = out.view([bs, -1, h, w]).type(in_type) / 255.
    return out

# random cutout
# TODO: should mask this 

def random_cutout(imgs, min_cut=10,max_cut=30):
    """
        args:
        imgs: np.array shape (B,C,H,W)
        min / max cut: int, min / max size of cutout 
        returns np.array
    """

    n, c, h, w = imgs.shape
    w1 = np.random.randint(min_cut, max_cut, n)
    h1 = np.random.randint(min_cut, max_cut, n)
    
    cutouts = np.empty((n, c, h, w), dtype=imgs.dtype)
    for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
        cut_img = img.copy()
        cut_img[:, h11:h11 + h11, w11:w11 + w11] = 0
        #print(img[:, h11:h11 + h11, w11:w11 + w11].shape)
        cutouts[i] = cut_img
    return cutouts

def random_cutout_color(imgs, min_cut=10,max_cut=30):
    """
        args:
        imgs: shape (B,C,H,W)
        out: output size (e.g. 84)
    """
    
    n, c, h, w = imgs.shape
    w1 = np.random.randint(min_cut, max_cut, n)
    h1 = np.random.randint(min_cut, max_cut, n)
    
    cutouts = np.empty((n, c, h, w), dtype=imgs.dtype)
    rand_box = np.random.randint(0, 255, size=(n, c)) / 255.
    for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
        cut_img = img.copy()
        
        # add random box
        cut_img[:, h11:h11 + h11, w11:w11 + w11] = np.tile(
            rand_box[i].reshape(-1,1,1),                                                
            (1,) + cut_img[:, h11:h11 + h11, w11:w11 + w11].shape[1:])
        
        cutouts[i] = cut_img
    return cutouts


def view_as_windows_cuda(x, window_shape):
	"""PyTorch CUDA-enabled implementation of view_as_windows"""
	assert isinstance(window_shape, tuple) and len(window_shape) == len(x.shape), \
		'window_shape must be a tuple with same number of dimensions as x'
	
	slices = tuple(slice(None, None, st) for st in torch.ones(4).long())
	win_indices_shape = [
		x.size(0),
		x.size(1)-int(window_shape[1]),
		x.size(2)-int(window_shape[2]),
		x.size(3)    
	]

	new_shape = tuple(list(win_indices_shape) + list(window_shape))
	strides = tuple(list(x[slices].stride()) + list(x.stride()))

	return x.as_strided(new_shape, strides)

# def random_mask1frame(x):
# 	n,c,h,w = x.shape 
# 	i_frame=random.choices(population=[0,1,2], weights=None, k=1)[0]
# 	# print(i_frame)
# 	# print(x.shape)
# 	# import pdb 
# 	# pdb.set_trace()
# 	erasing_func = kornia.augmentation.RandomErasing((0.018, 0.018), (.1, 1/.8), p=0.5)
# 	for j in range(3):
# 		x[:,i_frame*3:i_frame*3+3,:,:] = erasing_func(x[:,i_frame*3:i_frame*3+3,:,:])
# 	return x

def random_mask1frame(x):


	n,c,h,w = x.shape 
	# i_frame=random.choices(population=[0,1,2], weights=None, k=1)[0]
	# print(i_frame)
	# print(x.shape)
	# import pdb 
	# pdb.set_trace()
	erasing_func = kornia.augmentation.RandomErasing((0.03, 0.03), (.1, 1/.8), p=0.5)
	for i_frame in range(3):
		for j in range(3):
			x[:,i_frame*3:i_frame*3+3,:,:] = erasing_func(x[:,i_frame*3:i_frame*3+3,:,:])
			# p = random.uniform(0, 1)
			# if p > 0.5:
			# 	x[:,i_frame*3:i_frame*3+3,:,:] = x[:,i_frame*3:i_frame*3+3,:,:]
			# else:

			# 	x[:,i_frame*3:i_frame*3+3,:,:] = erasing_func(x[:,i_frame*3:i_frame*3+3,:,:])
	return x
def random_color_jitter(imgs):
    """
        inputs np array outputs tensor
    """
    b,c,h,w = imgs.shape
    imgs = imgs.view(-1,3,h,w)
    transform_module = nn.Sequential(ColorJitterLayer(brightness=0.4, 
                                                contrast=0.4,
                                                saturation=0.4, 
                                                hue=0.5, 
                                                p=1.0, 
                                                batch_size=128))

    imgs = transform_module(imgs).view(b,c,h,w)
    return imgs


# def random_mask1frame(x):


# 	n,c,h,w = x.shape 
# 	# i_frame=random.choices(population=[0,1,2], weights=None, k=1)[0]
# 	# print(i_frame)
# 	# print(x.shape)
# 	# import pdb 
# 	# pdb.set_trace()
# 	x0,x1,x2 = torch.chunk(x,3,dim=1)
# 	x = torch.cat((x0.unsqueeze(1),x1.unsqueeze(1),x2.unsqueeze(1)),dim=1)	
# 	# erasing_func = kornia.augmentation.VideoSequential(kornia.augmentation.RandomErasing((0.018, 0.018), (.1, 1/.8), p=0.5, data_format="BTCHW", same_on_frame=True,same_on_batch = False))


# 	erasing_func = kornia.augmentation.VideoSequential(
# 	     kornia.augmentation.RandomErasing((0.02, 0.02), (.1, 1/.8), p=0.5),
# 	     # kornia.color.BgrToRgb(),
# 	     # kornia.augmentation.RandomAffine(360, p=1.0),
# 	     # random_apply=10,
# 	     data_format="BTCHW",
# 	     same_on_frame=True,)
# 	# p = random.uniform(0, 1)

# 	# if p > 0.5:
# 	# 	x = erasing_func(x)
# 	x = erasing_func(x)
# 	# for i in range(n):
# 	# 	if p > 0.5:
# 	# 		x[i,:, :,:,:] = erasing_func(x[i,:, :,:,:].unsqueeze(0))
	

# 	x0,x1,x2 = torch.chunk(x,3,dim=1)
# 	x = torch.cat((x0.squeeze(1),x1.squeeze(1),x2.squeeze(1)),dim=1)
# 	# print(x.shape)
# 	# x = 
# 	# for i_frame in range(3):
# 	# 	for j in range(1):
# 	# 		p = random.uniform(0, 1)
# 	# 		if p > 0.5:
# 	# 			x[:,i_frame*3:i_frame*3+3,:,:] = x[:,i_frame*3:i_frame*3+3,:,:]
# 	# 		else:

# 	# 			x[:,i_frame*3:i_frame*3+3,:,:] = erasing_func(x[:,i_frame*3:i_frame*3+3,:,:])
# 	return x


def random_crop_2(imgs):
    output_size = imgs.shape

    n1 = imgs.shape[0]
    n2 = imgs.shape[1]
    n = n1 * n2
    imgs = imgs.reshape(n, *imgs.shape[2:])
    img_size = imgs.shape[-1]
    crop_max = img_size - output_size
    imgs = np.transpose(imgs, (0, 2, 3, 1))
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    windows = view_as_windows(
        imgs, (1, output_size[0], output_size[1], 1))[..., 0, :, :, 0]
    cropped_imgs = windows[np.arange(n), w1, h1]
    return cropped_imgs.reshape(n1, n2, *cropped_imgs.shape[1:])



# The SRM code, version 2, squared-ring shaped mask
# what is squared-ring shape: area between a big rectangle and a smaller rectangle
# this also works well compared with cirle-shaped ring, and is simpler to calculate
def random_mask_freq(x):

        p = random.uniform(0, 1)
        if p > 0.5:
         return x

        # dynamicly select freq range to erase
        A=0
        B=0.5
        a=random.uniform(A,B)
        C=2
        freq_limit_low = round(a,C)

        A=0
        B=0.05
        a=random.uniform(A,B)
        C=2
        diff = round(a,C)         
        freq_limit_hi = freq_limit_low + diff

        # b, 9, h, w
        b,c,h,w = x.shape
        x0,x1,x2 = torch.chunk(x, 3, dim=1)
        # b, 3, 3, h, w
        x = torch.cat((x0.unsqueeze(1),x1.unsqueeze(1),x2.unsqueeze(1)),dim=1)

        pass1 = torch.abs(torch.fft.fftfreq(x.shape[-1],device=x.device)) < freq_limit_hi
        pass2 = torch.abs(torch.fft.fftfreq(x.shape[-2],device=x.device)) < freq_limit_hi
        kernel1 = torch.outer(pass2, pass1) # freq_limit_hi square is true

        pass1 = torch.abs(torch.fft.fftfreq(x.shape[-1],device=x.device)) < freq_limit_low
        pass2 = torch.abs(torch.fft.fftfreq(x.shape[-2],device=x.device)) < freq_limit_low
        kernel2 = torch.outer(pass2, pass1) # freq_limit_low square is true

        kernel = kernel1 * (~kernel2)  # a square ring is true        
        fft_1 = torch.fft.fftn(x, dim=(2,3,4))
        imgs = torch.fft.ifftn(fft_1 * (~kernel), dim=(2,3,4)).float()
        x0,x1,x2 = torch.chunk(imgs,3,dim=1)
        imgs = torch.cat((x0.squeeze(1),x1.squeeze(1),x2.squeeze(1)),dim=1)

        return imgs


# The SRM code, version 1, circle-ring shaped mask
def random_mask_freq_v1(x):
        p = random.uniform(0, 1)
        if p > 0.5:
             return x
        # need to adjust r1 r2 and delta for best performance
        r1=random.uniform(0,0.5)
        delta_r=random.uniform(0,0.035)
        r2=np.min((r1+delta_r,0.5))
        # print(r2)
        # generate Mask M
        B,C,H,W = x.shape
        center = (int(H/2), int(W/2))
        diagonal_lenth = max(H,W) # np.sqrt(H**2+W**2) is also ok, use a smaller r1
        r1_pix = diagonal_lenth * r1
        r2_pix = diagonal_lenth * r2
        Y_coord, X_coord = np.ogrid[:H, :W]
        dist_from_center = np.sqrt((Y_coord - center[0])**2 + (X_coord - center[1])**2)
        M = dist_from_center <= r2_pix
        M = M * (dist_from_center >= r1_pix)
        M = ~M

        # mask Fourier spectrum
        M = torch.from_numpy(M).float().to(x.device)
        srm_out = torch.zeros_like(x)
        for i in range(C):
            x_c = x[:,i,:,:]
            x_spectrum = torch.fft.fftn(x_c, dim=(-2,-1))
            x_spectrum = torch.fft.fftshift(x_spectrum, dim=(-2,-1))
            out_spectrum = x_spectrum * M
            out_spectrum = torch.fft.ifftshift(out_spectrum, dim=(-2,-1))
            srm_out[:,i,:,:] = torch.fft.ifftn(out_spectrum, dim=(-2,-1)).float()
        return srm_out