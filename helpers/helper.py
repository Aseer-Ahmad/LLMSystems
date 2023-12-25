#helper.py
import torch
import torch
from torch.ao.quantization import (
  get_default_qconfig_mapping,
  get_default_qat_qconfig_mapping,
  QConfigMapping,
)
import torch.ao.quantization.quantize_fx as quantize_fx
import copy

import psutil
import os

def check_gpu_memory():
	gpu_mem = torch.cuda.memory_allocated() / 1e6
	gpu_mem_max = torch.cuda.max_memory_allocated() / 1e6

	print(f"GPU Memory Allocated: {gpu_mem} MB")
	print(f"GPU Max Memory Allocated: {gpu_mem_max} MB")
	return gpu_mem, gpu_mem_max

def check_cpu_memory():
	cpu_memory = psutil.virtual_memory()
	cpu_mem    = cpu_memory.used / 1e6

	print(f"CPU Memory Used: {cpu_mem} MB")
	return cpu_mem


def save_checkpoint(model, optimizer, lr_scheduler, checkpoint_path ):
	
	torch.save({
        'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'lr_state_dict' : lr_scheduler.state_dict()
    }, checkpoint_path)

	#report checkpoint size on-disk
	size_in_bytes = os.path.getsize(checkpoint_path)
	print(f"{checkpoint_path} : {size_in_bytes} bytes")

	return size_in_bytes


def load_checkpoint(model, optimizer, lr_scheduler, checkpoint_path):
	checkpoint = torch.load(checkpoint_path)
	model.load_state_dict(checkpoint['model_state_dict'])

	if optimizer != None and lr_scheduler != None :
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		lr_scheduler.load_state_dict(checkpoint['lr_state_dict'])
	
	return model, optimizer, lr_scheduler

def FX_graph_mode_quantization(model, input):

	# post training static quantization
	example_batch = (input)
	model_to_quantize = copy.deepcopy(model)
	qconfig_mapping = get_default_qconfig_mapping("qnnpack")
	model_to_quantize.eval()
	# calibrate : pass through model_prepared example batch
	# model(**example_batch)
	model_prepared = quantize_fx.prepare_fx(model_to_quantize, qconfig_mapping, example_batch)
	model_quantized = quantize_fx.convert_fx(model_prepared)

	return model_quantized

def eager_mode_quantization(model):
	pass

def metric1():
	pass
	
def metric2():
	pass
	
def getMemInfo():
	pass
	