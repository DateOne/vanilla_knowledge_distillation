import os
import shutil
from collections import OrderedDict

import json
import logging
from io import BytesIO

import numpy as np
import scipy.misc

import torch
import tensorflow as tf

class Params():
	'''class that load hyperparameters from a json file
	'''
	def __init__(self, json_path):
		with open(json_path) as f:
			params = json.load(f)
			self.__dict__.update(params)

	def save(self, json_path):
		with open(json_path, 'w') as f:
			json.dump(self.__dict__, f, indent=4)

	def update(self, json_path):
		with open(json_path) as f:
			params = json.load(f)
			self.__dict__.update(params)

	@property
	def dict(self):
		return self.__dict__

class RunningAverage():
	'''class to maintain an average of a quantity
	'''
	def __init__(self):
		self.steps = 0
		self.total = 0

	def update(self, val):
		self.total += val
		self.steps += 1

	def call(self):
		return self.total / float(self.steps)

def set_logger(log_path):
	'''funtion to set logger to log into terminal and file
	'''
	logger = logging.getLogger()
	logger.setLevel(logging.INFO)

	if not logger.handlers:
		file_handler = logging.FileHandler(log_path)
		file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
		logger.addHandler(file_handler)

		stream_handler = logging.StreamHandler()
		stream_handler.setFormatter(logging.Formatter('%(message)s'))
		logger.addHandler(stream_handler)

def save_dict_to_json(d, json_path):
	with open(json_path, 'w') as f:
		d = {k: float(v) for k, v in d.items()}
		json.dump(d, f, indent=4)

def save_checkpoint(state, is_best, checkpoint):
	'''save model and training parameters
	'''
	filepath = os.path.join(checkpoint, 'last.pth.tar')
	if not os.path.exists(checkpoint):
		print('checkpoint directory does not exist! Making directory {}'.format(checkpoint))
		os.mkdir(checkpoint)
	else:
		print('checkpoint directory exists!')
	
	torch.save(state, filepath)

	if is_best:
		shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))

def load_checkpoint(checkpoint, model, optimizer=None):
	'''load model and parameters
	'''
	if torch.cuda.is_available():
		checkpoint = torch.load(checkpoint)
	else:
		checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)

	model.load_state_dict(checkpoint['state_dict'])

	if optimizer:
		optimizer.load_state_dict(checkpoint['optim_dict'])

	return checkpoint

class Board_Logger(object):
	'''tensorboard log utility
	'''
	def __init__(self, log_dir):
		self.writer = tf.summary.FileWriter(log_dir)

	def scalar_summary(self, tag, images, step):
		img_summaries = []

		for i, img in enumerate(images):
			try:
				s = StringIO()
			except:
				s = BytesIO()

			scipy.misc.toimage(img).save(s, format='png')

			img_sum = tf.Summary.Image(
				encodeed_image_string=s.getvalue(),
				height=img.shape[0],
				width=img.shape[1])
			img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

		summary = tf.Summary(value=img_summaries)
		self.writer.add_summary(summary, step)

	def histo_summary(self, tag, values, step, bins=1000):
		counts, bin_edges = np.histogram(values, bins=bins)

		hist = tf.HistogramProto()
		hist.min = float(np.min(values))
		hist.max = float(np.max(values))
		hist.num = int(np.prod(values.shape))
		hist.sum = float(np.sum(values))
		hist.sum_squares = float(np.sum(values**2))

		bin_edges = bin_edges[1:]

		for edge in bin_edges:
			hist.bucket_limit.append(edge)
		for c in counts:
			hist.bucket.append(c)

		summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
		self.writer.add_summary(summary, step)
		self.writer.flush()