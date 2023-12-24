#helper.py
import torch
import psutil

def metric1():
	pass
	
def metric2():
	pass
	
def getMemInfo():
	pass
	
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