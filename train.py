#train.py
from dataloader import getDataset, getDataloaders
from helpers.helper import check_cpu_memory, check_gpu_memory, save_checkpoint, load_checkpoint

import yaml
import os
import time

# import tensorflow as tf

from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from torch.optim import AdamW, Adam, SGD
from transformers import get_scheduler
# import evaluate

from tqdm.auto import tqdm

import torch
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler 

import psutil


log_dir = "logs"  # Specify the directory where you want to store the logs
# summary_writer = tf.summary.create_file_writer(log_dir)

YAML_PATH = 'config.yaml'
PARENT_PATH  = os.getcwd()

def train(train_dataloader, trained_model_filename, yaml_data):

	print("\nstarting def train")
	#arguments
	MODEL_NAME        = yaml_data['MODEL_NAME']
	NUM_EPOCHS        = int(yaml_data['NUM_EPOCHS'])
	LR         	      = float(yaml_data['LR'])
	SAVE_CHKPNT_EPOCH = yaml_data['SAVE_CHKPNT_EPOCH']
	MODEL_CHKPNT_DIR  = yaml_data['MODEL_CHKPNT_DIR']
	SEQ_LEN           = int(yaml_data['SEQ_LEN'])
	BATCH_SIZE		  = int(yaml_data['BATCH_SIZE'])
	OPTIMIZER_NAME    = yaml_data['OPTIMIZER_NAME']
	OPT_LVL 		  = yaml_data['OPT_LEVEL']
	PRECISION_TYPE    = yaml_data['PRECISION_TYPE']

	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	
	print(f"MODEL_NAME : {MODEL_NAME}\nNUM_EPOCHS : {NUM_EPOCHS} \nLR : {LR}\nSAVE_CHKPNT_EPOCH : {SAVE_CHKPNT_EPOCH} \
	   MODEL_CHKPNT_DIR : {MODEL_CHKPNT_DIR}\nSEQ_LEN : {SEQ_LEN}\nBATCH_SIZE : {BATCH_SIZE}\nOPTIMIZER_NAME : {OPTIMIZER_NAME}\ndevice : {device}\nOPT_LEVEL : {OPT_LVL}\n\
	   PRECISION_TYPE:{PRECISION_TYPE}\n")
	
	num_batches = len(train_dataloader)
	step = 0
	epoch_total_time  = 0
	forward_total_time, backward_total_time = 0, 0
	running_loss = 0

	print(f"\nloading model {MODEL_NAME} , optimizer and scheduler")

	model = AutoModelForCausalLM.from_pretrained(MODEL_NAME) # , low_cpu_mem_usage = True
	model.to(device)
	num_training_steps = NUM_EPOCHS * num_batches
	optimizer    = get_opt(model, OPTIMIZER_NAME, yaml_data)
	lr_scheduler = get_schdlr(optimizer, num_training_steps)
	scaler       = GradScaler(enabled = True)
	
	# #amp initialization
	# #avoiding as outputs.loss has unexpected behaviour 
	# model, optimizer = amp.initialize(model, optimizer, opt_level = OPT_LVL)

	if trained_model_filename != None:
		model_chkpnt = os.path.join(yaml_data['MODEL_CHKPNT_DIR'], f'{trained_model_filename}.pth')  
		model, optimizer, lr_scheduler = load_checkpoint(model, optimizer, lr_scheduler, model_chkpnt)	
		print(f'{MODEL_NAME} loaded from {model_chkpnt}')

	model.train()
	
	# progress_bar = tqdm(range(num_training_steps))
	print("\nmodel, opt, schdl loaded")
	print("\nbeginning training ...")
	
	for epoch in range(NUM_EPOCHS):
		
		start_time = time.time()

		try : 
			for i, batch in enumerate(train_dataloader):
				
				batch = {k: v.to(device) for k, v in batch.items()}

				if PRECISION_TYPE == 'MIXED':
					with autocast(dtype = torch.float16, enabled = True):
						forward_st = time.time()
						outputs = model(**batch)
						forward_et = time.time()
						forward_total_time += (forward_et - forward_st)
				elif PRECISION_TYPE == 'SINGLE':
						forward_st = time.time()
						outputs = model(**batch)
						forward_et = time.time()
						forward_total_time += (forward_et - forward_st)

				loss = outputs.loss

				running_loss += loss

				# #avoiding as outputs.loss has unexpected behaviour 
				# with amp.scale_loss(loss, optimizer) as scaled_loss:
				# 	scaled_loss.backward()

				backward_st = time.time()
				
				if PRECISION_TYPE == 'MIXED':
					scaler.scale(loss).backward() 
				elif PRECISION_TYPE == 'SINGLE':
					loss.backward()
				
				backward_et = time.time()
				backward_total_time += (backward_et - backward_st)

				if PRECISION_TYPE == 'MIXED':
					scaler.step(optimizer)
					scale = scaler.get_scale()
					scaler.update()
					skip_lr_shdlr = (scale > scaler.get_scale()) # to avoid lr step in case of NaN gradients

					if not skip_lr_shdlr : 
						lr_scheduler.step()
				
				elif PRECISION_TYPE == 'SINGLE':
					optimizer.step()
					lr_scheduler.step()

				optimizer.zero_grad()
				
				# progress_bar.update(1)

				step += 1

				print(f"epoch : {epoch+1} / {NUM_EPOCHS} iter : {i} / {num_batches},  loss : {loss}")
				# tf.summary.scalar('loss', loss, step = step)

				# gpu_mem, gpu_mem_max = check_gpu_memory()
				# cpu_mem              = check_cpu_memory()
				# tf.summary.scalar('gpu_mem', gpu_mem, step = step)
				# tf.summary.scalar('gpu_mem_max', gpu_mem_max, step = step)
				# tf.summary.scalar('cpu_mem', cpu_mem, step = step)
		
		except RuntimeError as e:
			print(f"RuntimeError : {e}")

		end_time = time.time()
		epoch_time = end_time - start_time
		epoch_total_time += epoch_time

		#average training loss
		print(f"epoch : {epoch+1} / {NUM_EPOCHS} iter : {i} / {num_batches}, average training loss : {running_loss / num_batches}")
		running_loss = 0
		
		#training time per epoch
		print(f'epoch : {epoch+1} total time : {epoch_time:.2f} seconds')
		# tf.summary.scalar('epoch_exe_time', epoch_time, step = epoch)

		#throughput : tokens processed per second
		t_tps = epoch_time / (SEQ_LEN * BATCH_SIZE * num_batches)
		print(f'epoch : {epoch+1} tokens processed per second : {t_tps:.4f} seconds')
		# tf.summary.scalar('token_throughput', t_tps, step = epoch)

		check_gpu_memory()
		check_cpu_memory()

		print(f"lr schdl : {lr_scheduler.state_dict()}")

		#save checkpoint on disk 
		if SAVE_CHKPNT_EPOCH is not None and epoch % SAVE_CHKPNT_EPOCH == 0:
			checkpoint_path = os.path.join(PARENT_PATH, MODEL_CHKPNT_DIR, f'{MODEL_NAME}_chkpoint_{epoch+1}.pth')
			save_checkpoint(model, optimizer, lr_scheduler, checkpoint_path)

	#total training time per epoch
	print(f'Total training Time for {NUM_EPOCHS} epoch : {epoch_total_time :.2f} seconds')

	#average training time per epoch
	print(f'Average training Time per epoch : {epoch_total_time / NUM_EPOCHS :.2f} seconds')

	#average forward pass time per epoch
	print(f'Average forward pass Time per epoch : {forward_total_time / NUM_EPOCHS :.2f} seconds')

	#average backward pass time per epoch
	print(f'Average backward pass Time per epoch : {backward_total_time / NUM_EPOCHS :.2f} seconds')

	return model

def get_opt(model, OPTIMIZER_NAME, yaml_data):

	#arguments
	LR           = float(yaml_data['LR'])
	WEIGHT_DECAY = float(yaml_data['WEIGHT_DECAY'])
	MOMENTUM     = float(yaml_data['MOMENTUM'])

	if OPTIMIZER_NAME == 'AdamW':
		optimizer = AdamW(model.parameters(), lr=LR, weight_decay = WEIGHT_DECAY)
	elif OPTIMIZER_NAME == 'Adam':
		optimizer = Adam(model.parameters(), lr=LR, weight_decay = WEIGHT_DECAY)
	elif OPTIMIZER_NAME == 'SGD':
		optimizer = SGD(model.parameters(), lr=LR, momentum = MOMENTUM)

	return optimizer

def get_schdlr(optimizer, num_training_steps):
	
	lr_scheduler = get_scheduler(
    	name="linear", optimizer=optimizer,
		num_warmup_steps=0, 
		num_training_steps=num_training_steps
	)

	return lr_scheduler

def eval(model, eval_dataloader, model_chkpnt, yaml_data):
	
	#arguments
	MODEL_NAME        = yaml_data['MODEL_NAME']

	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	metric = None#evaluate.load("accuracy")	

	if model_chkpnt != None:
		model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
		model = load_checkpoint(model, model_chkpnt)

	model.eval()

	for batch in eval_dataloader:
		batch = {k: v.to(device) for k, v in batch.items()}
		with torch.no_grad():
			outputs = model(**batch)

	logits = outputs.logits
	predictions = torch.argmax(logits, dim=-1)
	metric.add_batch(predictions=predictions, references=batch["labels"])

	metric.compute()

def config():
    # Read the YAML file
    with open(YAML_PATH, 'r') as file:
        yaml_data = yaml.safe_load(file)

    return yaml_data

def free_memory():
	torch.cuda.empty_cache()

def loadModel(yaml_data):
	MODEL_NAME        = yaml_data['MODEL_NAME']
	print(f"\nloading model {MODEL_NAME}")
	model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, low_cpu_mem_usage = True)
	return model

def main():
	yaml_data  = config()
	print(yaml_data)

	# set trained_model_filename if need to use a pretrained checkpoint ; else keep None
	trained_model_filename = None

	# load dataset
	data = getDataset(yaml_data)
	train_dataloader, eval_dataloader = getDataloaders(data, yaml_data)
	model = train(train_dataloader, trained_model_filename,  yaml_data)
	
	# eval(model, eval_dataloader, trained_model_filename, yaml_data)

if __name__ == '__main__':
	main()
