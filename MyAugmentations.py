import torch
from torchvision import transforms
from torchvision.transforms import Compose
import cv2
import numpy as np
import types
from numpy import random

class TrainAugmentation(object):
	def  __init__(self, size=224, _mean=[0.34358176, 0.38097752, 0.36801731],
		_std=[0.18488177, 0.18543289, 0.20352854] ):

		self.augment = Compose([
			RandomCrop(size),
			RandomMirror(),# 随机左右翻转
			Normalize(mean = _mean, std = _std),
			ToTensor(),
		])

	def __call__(self, img):
		return self.augment(img)

class TestAugmentation(object):
	def  __init__(self, size=224, _mean=[0.34358176, 0.38097752, 0.36801731],
		_std=[0.18488177, 0.18543289, 0.20352854] ):

		self.augment = Compose([
            CenterCrop(size),
			Normalize(mean = _mean, std = _std),
			ToTensor(),
		])

	def __call__(self, img):
		return self.augment(img)


class Resize(object):
	def __init__(self, size):
		self.size = size

	def __call__(self, image):
		return cv2.resize(image, (self.size, self.size))

class Normalize(object):
	def __init__(self, mean, std):
		self.mean = np.array(mean, dtype=np.float32)
		self.std = np.array(std, dtype=np.float32)

	def __call__(self, image):
		image = image.astype(np.float32) / 255
		image[:,:] -= self.mean
		image[:,:] /= self.std
		return image.astype(np.float32)


class ConvertFromInts(object):
	def __call__(self, image):
		return image.astype(np.float32)


class ExpandRandomCrop(object):
	def __init__(self, mean, expand_ratio=0.125):
		self.mean = mean
		self.expand_ratio = expand_ratio

	def __call__(self, image):
		if random.randint(2):
			return image

		ratio = self.expand_ratio
		height, width, depth = image.shape
		left = random.uniform(0, width * 2*ratio)
		top = random.uniform(0, height * 2*ratio)

		expand_image = np.zeros(
			(int(height * (2 * ratio + 1)), int(width * (2 * ratio + 1)), depth),
			dtype=image.dtype)
		expand_image[:, :, :] = self.mean
		expand_image *= 255

		# put original image into the center of expand image
		expand_image[int(height * ratio):int(height * ratio + height),
					 int(width * ratio):int(width * ratio + width)] = image


		image = expand_image[int(top):int(top + height),
					 int(left):int(left + width)]

		return image


class RandomCrop(object):
	def __init__(self, size=224):
		self.size = size

	def __call__(self, image):
		size = self.size

		if random.randint(2):
			return cv2.resize(image,(size,size))

		# ratio = self.expand_ratio
		height, width, depth = image.shape
		left = random.uniform(0, width - size)
		top = random.uniform(0, height - size)

		crop_image = image[int(top):int(top + size),
					 int(left):int(left + size)]

		return crop_image

class CenterCrop(object):
	def __init__(self, size=224):
		self.size = size

	def __call__(self, image):
		size = self.size

		# ratio = self.expand_ratio
		height, width, depth = image.shape
		left = (width - size) / 2
		top = (height - size) / 2

		crop_image = image[int(top):int(top + size),
					 int(left):int(left + size)]

		return crop_image

class RandomMirror(object):
	def __call__(self, image):
		if random.randint(2):
			image = image[:, ::-1]

		return image


class RandomSaturation(object):
	def __init__(self, lower=0.5, upper=1.5):
		self.lower = lower
		self.upper = upper
		assert self.upper >= self.lower, "contrast upper must be >= lower."
		assert self.lower >= 0, "contrast lower must be non-negative."

	def __call__(self, image):
		if random.randint(2):
			image[:, :, 1] *= random.uniform(self.lower, self.upper) #image.shape = (h,w,ch)

		return image


class RandomHue(object):
	def __init__(self, delta=18.0):
		assert delta >= 0.0 and delta <= 360.0
		self.delta = delta

	def __call__(self, image):
		if random.randint(2):
			image[:, :, 0] += random.uniform(-self.delta, self.delta)
			image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
			image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
		return image


class RandomLightingNoise(object):
	def __init__(self):
		self.perms = ((0, 1, 2), (0, 2, 1),
					  (1, 0, 2), (1, 2, 0),
					  (2, 0, 1), (2, 1, 0))

	def __call__(self, image):
		if random.randint(2):
			swap = self.perms[random.randint(len(self.perms))]
			shuffle = SwapChannels(swap)  # shuffle channels
			image = shuffle(image)
		return image


class ConvertColor(object):
	def __init__(self, current='BGR', transform='HSV'):
		self.transform = transform
		self.current = current

	def __call__(self, image):
		if self.current == 'BGR' and self.transform == 'HSV':
			image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		elif self.current == 'HSV' and self.transform == 'BGR':
			image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
		else:
			raise NotImplementedError
		return image


class RandomContrast(object):
	def __init__(self, lower=0.5, upper=1.5):
		self.lower = lower
		self.upper = upper
		assert self.upper >= self.lower, "contrast upper must be >= lower."
		assert self.lower >= 0, "contrast lower must be non-negative."

	# expects float image
	def __call__(self, image):
		if random.randint(2):
			alpha = random.uniform(self.lower, self.upper)
			image *= alpha
		return image


class RandomBrightness(object):
	def __init__(self, delta=32):
		assert delta >= 0.0
		assert delta <= 255.0
		self.delta = delta

	def __call__(self, image):
		if random.randint(2):
			delta = random.uniform(-self.delta, self.delta)
			image += delta
		return image


class ToCV2Image(object):
	def __call__(self, tensor):
		return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0))

class ToTensor(object):
	def __call__(self, cvimage):
		return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1)


class SwapChannels(object):
	"""Transforms a tensorized image by swapping the channels in the order
	 specified in the swap tuple.
	Args:
		swaps (int triple): final order of channels
			eg: (2, 1, 0)
	"""

	def __init__(self, swaps):
		self.swaps = swaps

	def __call__(self, image):
		"""
		Args:
			image (Tensor): image tensor to be transformed
		Return:
			a tensor with channels swapped according to swap
		"""
		# if torch.is_tensor(image):
		#     image = image.data.cpu().numpy()
		# else:
		#     image = np.array(image)
		image = image[:, :, self.swaps]
		return image


class PhotometricDistort(object):
	def __init__(self):
		self.pd = [
			RandomContrast(),# 对比度随机调整，img*alpha
			ConvertColor(transform='HSV'),
			RandomSaturation(),# 饱和度随机调整，img*alpha
			RandomHue(),
			ConvertColor(current='HSV', transform='BGR'),
			RandomContrast()# 对比度随机调整，img*alpha
		]
		self.rand_brightness = RandomBrightness()# 亮度随机调整，img + alpha
		self.rand_light_noise = RandomLightingNoise()# 通道随机交换

	def __call__(self, image):
		im = image.copy()
		im = self.rand_brightness(im)# 亮度必调整
		if random.randint(2):# 对比度调整一次或两次
			distort = Compose(self.pd[:-1])
		else:
			distort = Compose(self.pd[1:])
		im = distort(im)
		return self.rand_light_noise(im)# 通道必随机交换
