from __future__ import division, print_function
from PIL import Image
from functools import partial
import os
import sys
import tensorflow as tf
import torchvision.utils as vutils
import numpy as np
import time
import math
import copy
import scipy.misc as spm
import scipy.ndimage as spi
import scipy.sparse as sps
import torch

try:
	xrange          # Python 2
except NameError:
	xrange = range  # Python 3

VGG_MEAN = [103.939, 116.779, 123.68]

def getlaplacian1(i_arr, consts, epsilon=1e-5, win_rad=1):
	neb_size = (win_rad * 2 + 1) ** 2
	h, w, c = i_arr.shape
	img_size = w * h
	consts = spi.morphology.grey_erosion(consts, footprint=np.ones(shape=(win_rad * 2 + 1, win_rad * 2 + 1)))

	indsM = np.reshape(np.array(range(img_size)), newshape=(h, w), order='F')
	tlen = int((-consts[win_rad:-win_rad, win_rad:-win_rad] + 1).sum() * (neb_size ** 2))
	row_inds = np.zeros(tlen)
	col_inds = np.zeros(tlen)
	vals = np.zeros(tlen)
	l = 0
	for j in range(win_rad, w - win_rad):
		for i in range(win_rad, h - win_rad):
			if consts[i, j]:
				continue
			win_inds = indsM[i - win_rad:i + win_rad + 1, j - win_rad: j + win_rad + 1]
			win_inds = win_inds.ravel(order='F')
			win_i = i_arr[i - win_rad:i + win_rad + 1, j - win_rad: j + win_rad + 1, :]
			win_i = win_i.reshape((neb_size, c), order='F')
			win_mu = np.mean(win_i, axis=0).reshape(c, 1)
			win_var = np.linalg.inv(
				np.matmul(win_i.T, win_i) / neb_size - np.matmul(win_mu, win_mu.T) + epsilon / neb_size * np.identity(
					c))

			win_i2 = win_i - np.repeat(win_mu.transpose(), neb_size, 0)
			tvals = (1 + np.matmul(np.matmul(win_i2, win_var), win_i2.T)) / neb_size

			ind_mat = np.broadcast_to(win_inds, (neb_size, neb_size))
			row_inds[l: (neb_size ** 2 + l)] = ind_mat.ravel(order='C')
			col_inds[l: neb_size ** 2 + l] = ind_mat.ravel(order='F')
			vals[l: neb_size ** 2 + l] = tvals.ravel(order='F')
			l += neb_size ** 2

	vals = vals.ravel(order='F')[0: l]
	row_inds = row_inds.ravel(order='F')[0: l]
	col_inds = col_inds.ravel(order='F')[0: l]
	a_sparse = sps.csr_matrix((vals, (row_inds, col_inds)), shape=(img_size, img_size))

	sum_a = a_sparse.sum(axis=1).T.tolist()[0]
	a_sparse = sps.diags([sum_a], [0], shape=(img_size, img_size)) - a_sparse

	return a_sparse

def getLaplacian(img):
	h, w, _ = img.shape
	coo_mat = getlaplacian1(img, np.zeros(shape=(h, w)), 1e-5, 1).tocoo()
	values = coo_mat.data
	indices = np.vstack((coo_mat.row, coo_mat.col))
	i = torch.LongTensor(indices)
	v = torch.FloatTensor(values)
	shape = coo_mat.shape
	sparse_mat = torch.sparse.FloatTensor(i, v, torch.Size(shape))
	return sparse_mat

def stylize(content_image_path):
	content_image = np.array(Image.open(content_image_path).convert("RGB"), dtype=np.float32)
	content_width, content_height = content_image.shape[1], content_image.shape[0]
	content_image = rgb2bgr(content_image)
	content_image = content_image.reshape((1, content_height, content_width, 3)).astype(np.float32)
	mean_pixel = torch.tensor(VGG_MEAN)
	init_image = np.random.randn(1, content_height, content_width, 3).astype(np.float32) * 0.0001
	input_image = torch.tensor(init_image)
	input_image_plus = torch.squeeze(input_image + mean_pixel, 0)
	return input_image_plus

def affine_loss(output, M, weight):
	loss_affine = 0.0
	output_t = output / 255.
	for Vc in torch.unbind(output_t, dim=-1):
		Vc_ravel = torch.reshape(torch.t(Vc), (-1,))
		loss_affine += torch.mm(torch.unsqueeze(Vc_ravel, 0), torch.mm(M, torch.unsqueeze(Vc_ravel, -1)))

	return loss_affine * weight
	
def rgb2bgr(rgb, vgg_mean=True):
	if vgg_mean:
		return rgb[:, :, ::-1] - VGG_MEAN
	else:
		return rgb[:, :, ::-1]

def get_affine_loss(content_image_path):
# 	best_result = stylize(content_image_path)
# 	a = best_result.numpy()
# 	result = Image.fromarray(np.uint8(np.clip(a[:, :, ::-1], 0, 255.0)))
 	init_image_path = "datasets/red_sky/trainA/00000.jpg"
 	content_image = np.array(Image.open(content_image_path).convert("RGB"), dtype=np.float32)
 	content_width, content_height = content_image.shape[1], content_image.shape[0]
 	M = getLaplacian(content_image / 255.)
 	content_image = rgb2bgr(content_image)
 	content_image = content_image.reshape((1, content_height, content_width, 3)).astype(np.float32)
 	#init_image = init_image = np.random.randn(1, content_height, content_width, 3).astype(np.float32) * 0.0001
 	init_image = np.expand_dims(rgb2bgr(np.array(Image.open(init_image_path).convert("RGB"), dtype=np.float32)).astype(np.float32), 0)
 	mean_pixel = torch.Tensor(VGG_MEAN)
 	input_image = torch.Tensor(init_image)
 	input_image_plus = torch.squeeze(input_image + mean_pixel, 0)
 	loss_affine = affine_loss(input_image_plus, M, 1e4)
 	return loss_affine