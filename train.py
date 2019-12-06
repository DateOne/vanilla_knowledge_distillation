import argparse
import os
import time
import math
import random
from tqdm import tqdm

import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

import utils
import model.data_loader as data_loader

import model.net as net
import model.resnet as resnet
import model.wrn as wrn
import model.densenet as densenet
import model.resnext as resnext
import model.preresnet as preresnet

from evaluate import evaluate, evaluate_kd

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")

def train(model, optimizer, loss_fn, dataloader, metrics, params):
	model.train()
	summ = []
	loss_avg = utils.RunningAverage()

	with tqdm(total=len(dataloader)) as t:
		for i, (train_batch, labels_batch) in enumerate(dataloader):
			if params.cuda:
				train.batch, labels_batch = train_batch.cuda(async=True), labels_batch.cuda(asyns=True)
			train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)
			output_batch = model(train_batch)
			loss = loss_fn(output_batch, labels_batch)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if i % params.save_summary_steps == 0:
				output_batch = output_batch.data.cpu().numpy()
				labels_batch = labels_batch.data.cpu().numpy()

				summary_batch = {metric:metrics[metric](output_batch, labels_batch) for metric in metrics}
				summary_batch['loss'] = loss.data[0]
				summ.append(summary_batch)

			loss_avg.update(loss.data[0])

			t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
			t.update()

	metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
	metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
	logging.info("- Train metrics: " + metrics_string)

def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fn, metrics, params, model_dir, restore_file=None):
	'''train and evaluate
	'''
	if restore_file is not None:
		restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
		logging.info("Restoring parameters from {}".format(restore_path))
		utils.load_checkpoint(restore_path, model, optimizer)

	best_val_acc = 0.0

	if params.model_version == "resnet18":
		scheduler = StepLR(optimizer, step_size=150, gamma=0.1)
	elif params.model_version == "cnn":
		scheduler = StepLR(optimizer, step_size=100, gamma=0.2)

	for epoch in range(params.num_epochs):
		scheduler.step()
		logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))
		train(model, optimizer, loss_fn, train_dataloader, metrics, params)
		val_metrics = evaluate(model, loss_fn, val_dataloader, metrics, params)
		val_acc = val_metrics['accuracy']
		is_best = val_acc>=best_val_acc
		utils.save_checkpoint(
			{
			'epoch': epoch + 1,
			'state_dict': model.state_dict(),
			'optim_dict' : optimizer.state_dict()},
			is_best=is_best,
			checkpoint=model_dir)
		if is_best:
			logging.info("- Found new best accuracy")
			best_val_acc = val_acc
			best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
			utils.save_dict_to_json(val_metrics, best_json_path)
			last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
			utils.save_dict_to_json(val_metrics, last_json_path)

def fetch_teacher_outputs(teacher_model, dataloader, params):
	teacher_model.eval()
	teacher_outputs = []
	for i, (data_batch, labels_batch) in enumerate(dataloader):
		if params.cuda:
			data_batch, labels_batch = data_batch.cuda(async=True), labels_batch.cuda(async=True)
			data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)
			output_teacher_batch = teacher_model(data_batch).data.cpu().numpy()
			teacher_outputs.append(output_teacher_batch)

			return teacher_outputs

def train_kd(model, teacher_outputs, optimizer, loss_fn_kd, dataloader, metrics, params):
	model.train()

	summ = []
	loss_avg = utils.RunningAverage()

	with tqdm(total=len(dataloader)) as t:
		for i, (train_batch, labels_batch) in enumerate(dataloader):
			if params.cuda:
				train_batch, labels_batch = train_batch.cuda(async=True), labels_batch.cuda(async=True)
			train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

			output_batch = model(train_batch)

			output_teacher_batch = torch.from_numpy(teacher_outputs[i])

			if params.cuda:
				output_teacher_batch = output_teacher_batch.cuda(async=True)
			output_teacher_batch = Variable(output_teacher_batch, requires_grad=False)

			loss = loss_fn_kd(output_batch, labels_batch, output_teacher_batch, params)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if i % params.save_summary_steps == 0:
				output_batch = output_batch.data.cpu().numpy()
				labels_batch = labels_batch.data.cpu().numpy()

				summary_batch = {metric:metrics[metric](output_batch, labels_batch) for metric in metrics}
				summary_batch['loss'] = loss.data[0]
				summ.append(summary_batch)

			loss_avg.update(loss.data[0])

			t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
			t.update()

	metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
	metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
	logging.info("- Train metrics: " + metrics_string)

def train_and_evaluate_kd(model, teacher_model, train_dataloader, val_dataloader, optimizer, loss_fn_kd, metrics, params, model_dir, restore_file=None):
	if restore_file is not None:
		restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
		logging.info("Restoring parameters from {}".format(restore_path))
		utils.load_checkpoint(restore_path, model, optimizer)

	best_val_acc = 0.0

	loading_start = time.time()
	teacher_model.eval()
	teacher_outputs = fetch_teacher_outputs(teacher_model, train_dataloader, params)
	elapsed_time = math.ceil(time.time() - loading_start)
	logging.info("- Finished computing teacher outputs after {} secs..".format(elapsed_time))

	if params.model_version == "resnet18_distill":
		scheduler = StepLR(optimizer, step_size=150, gamma=0.1)
	elif params.model_version == "cnn_distill":
		scheduler = StepLR(optimizer, step_size=100, gamma=0.2) 

	for epoch in range(params.num_epochs):
		scheduler.step()

		logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

		train_kd(model, teacher_outputs, optimizer, loss_fn_kd, train_dataloader, metrics, params)

		val_metrics = evaluate_kd(model, val_dataloader, metrics, params)

		val_acc = val_metrics['accuracy']
		is_best = val_acc>=best_val_acc

		utils.save_checkpoint(
			{
			'epoch': epoch + 1,
			'state_dict': model.state_dict(),
			'optim_dict' : optimizer.state_dict()},
			is_best=is_best,
			checkpoint=model_dir)

		if is_best:
			logging.info("- Found new best accuracy")
			best_val_acc = val_acc

			best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
			utils.save_dict_to_json(val_metrics, best_json_path)

		last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
		utils.save_dict_to_json(val_metrics, last_json_path)

if __name__ == '__main__':
	args = parser.parse_args()
	json_path = os.path.join(args.model_dir, 'params.json')
	assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
	params = utils.Params(json_path)

	params.cuda = torch.cuda.is_available()

	random.cuda = torch.cuda.is_available()

	random.seed(230)
	torch.manual_seed(230)

	if params.cuda: torch.cuda.manual_seed(230)

	utils.set_logger(os.path.join(args.model_dir, 'train.log'))

	logging.info('Loading the datasets...')

	if params.subset_persent < 1.0:
		train_dl = data_loader.fetch_subset_dataloader('train', params)
	else:
		train_dl = data_loader.fetch_dataloader('train', params)

	dev_dl = data_loader.fetch_dataloader('dev', params)

	logging.info('- done.')

	if 'distill' in params.model_version:
		if params.model_version == 'cnn_distill':
			model = net.Net(params).cuda() if params.cuda else net.Net(params)
			optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
			loss_fn_kd = net.loss_fn_kd
			metrics = net.metrics
		elif params.model_version == 'resnet18_distill':
			model = resnet.ResNet18().cuda() if params.cuda else resnet18()
			optimizer = optim.SGD(model.parameters(), lr=params.learning_rate, momentum=0.9, weight_decay=5e-4)
			loss_fn_kd = net.loss_fn_kd
			metrics = resnet.metrics

		if params.teacher == "resnet18":
			teacher_model = resnet.ResNet18()
			teacher_checkpoint = 'experiments/base_resnet18/best.pth.tar'
			teacher_model = teacher_model.cuda() if params.cuda else teacher_model
		elif params.teacher == "wrn":
			teacher_model = wrn.WideResNet(depth=28, num_classes=10, widen_factor=10, dropRate=0.3)
			teacher_checkpoint = 'experiments/base_wrn/best.pth.tar'
			teacher_model = nn.DataParallel(teacher_model).cuda()
		elif params.teacher == "densenet":
			teacher_model = densenet.DenseNet(depth=100, growthRate=12)
			teacher_checkpoint = 'experiments/base_densenet/best.pth.tar'
			teacher_model = nn.DataParallel(teacher_model).cuda()
		elif params.teacher == "resnext29":
			teacher_model = resnext.CifarResNeXt(cardinality=8, depth=29, num_classes=10)
			teacher_checkpoint = 'experiments/base_resnext29/best.pth.tar'
			teacher_model = nn.DataParallel(teacher_model).cuda()
		elif params.teacher == "preresnet110":
			teacher_model = preresnet.PreResNet(depth=110, num_classes=10)
			teacher_checkpoint = 'experiments/base_preresnet110/best.pth.tar'
			teacher_model = nn.DataParallel(teacher_model).cuda()
		
		utils.load_checkpoint(teacher_checkpoint, teacher_model)

		logging.info("Experiment - model version: {}".format(params.model_version))
		logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
		logging.info("First, loading the teacher model and computing its outputs...")
		train_and_evaluate_kd(model, teacher_model, train_dl, dev_dl, optimizer, loss_fn_kd, metrics, params, args.model_dir, args.restore_file)

	else:
		if params.model_version == 'cnn':
			model =  net.Net(params).cuda() if params.cuda else resnet.ResNet18()
			optimizer = optim.SGD(model.parameters(), lr=params.learning_rate, momentum=0.9, weight_decay=5e-4)

			loss_fn = resnet.loss_fn
			metrics = resnet.metrics

		elif params.model_version == 'resnet18':
			model = resnet.ResNet18().cuda() if params.cuda else resnet.ResNet18()
			optimizer = optim.SGD(model.parameters(), lr=params.learning_rate, momentum=0.9, weight_decay=5e-4)

			loss_fn = resnet.loss_fn
			metrics = resnet.metrics

		logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
		train_and_evaluate(model, train_dl, dev_dl, optimizer, loss_fn, metrics, params, args.model_dir, args.restore_file)
