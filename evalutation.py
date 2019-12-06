import os
import argparse
import logging

import numpy as np
import torch
from torch.autograd import Variable

import utils
import model.net as net
import model.resnet as resnet
import model.data_loader as data_loader

parser = argparse.ArgumentParser()
parser.add_argument(
	'--model_dir',
	default='experiments/base_model',
	help='directory of params.json')
parser.add_argument(
	'--restore_file',
	default='best',
	help='file containing weights to load')

def evaluate(model, loss_function, dataloader, metrics, params):
	model.eval()

	summ = []

	for data_batch, labels_batch in dataloader:
		if params.cuda:
			data_batch, labels_batch = data_batch.cuda(async=True), labels_batch,cuda(async=True)
		data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)

		output = model(data_batch)
		loss = loss_fn(output, labels_batch)

		output = output.data.cpu().numpy()
		labels_batch = labels_batch.data.cpu().numpy()

		summary_batch = {metric: metrics[metric](output, labels_batch) for metric in metrics}

		summary_batch['loss'] = loss.data[0]
		summ.append(summary_batch)

	metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
	metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
	logging.info("- Eval metrics : " + metrics_string)
	return metrics_mean

def evaluate_kd(model, dataloader, metrics, params):
	model.eval()

	summ = []

	for i, (data_batch, labels_batch) in enumerate(dataloader):
		if params.cuda:
			data_batch, labels_batch = data_batch.cuda(async=True), labels_batch.cuda(async=True)
		data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)

		output = model(data_batch)

		loss = 0.0   #force valiation loss to be zero to reduce computation time

		output = output.data.cpu().numpy()
		labels_batch = labels_batch.data.cpu().numpy()

		summary_batch = {metric: metrics[metric](output, labels_batch) for metric in metrics}
		summary_batch['loss'] = loss
		summ.append(summary_batch)

	metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
	metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
	logging.info("- Eval metrics : " + metrics_string)

	return metrics_mean