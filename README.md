# LLMSystems

Rough Doc. : https://docs.google.com/document/d/1HLnIiyt8Fx_KEZq3ItU9e-tyhKzCqg4ra_8UUIhSKh4/edit<br>

<h4>Mixed-Precision Mini-Project</h4>

Completion requirements

Opened: Tuesday, 7 November 2023, 3:00 PM <br>
Due: Tuesday, 5 December 2023, 11:59 PM <br>
REPORT : https://docs.google.com/document/d/1qyORiiEgWv5z1f-hx-plwBSuunqNJ1jZsD1GPMEwmK0/edit?usp=sharing <br>

The goal of the mini-project is to get hands-on experience with mixed-precision training. You will fine-tune a model using both single- and mixed-precision.

Instructions:

1. Fine-tune a GPT2 model on the Databricks Dolly 15K dataset. Do so using both single-precision (FP32 or TF32) and mixed-precision (FP16 or BF16) weights in two separate training runs. <br>
2. For both the single- and mixed-precision runs, report:<br>
3. The training throughput in tokens per second and the average training time per epoch (i.e. per full iteration over the dataset).<br>
4. The memory consumption, both GPU memory as well as on-disk checkpoint size.<br>
5. The training loss on the dataset, after 0, 1, 2 and 3 epochs of training.<br>
6. Bonus: after training, convert the model to int8 format, in a way that preserves as much of its performance as possible, and report the metrics mentioned above for inference.<br>
7. Before the next lecture, hand in a report describing:<br>
8. Your approach to solve the problem, and in particular how you changed the model precision. You should also mention relevant tools and libraries that you used, and hyperparameter values that were important. The results, i.e. the evaluation metrics described above. The code you used to solve the assignment.<br>


USEFUL LINKS : 

1. https://huggingface.co/docs/transformers/model_doc/gpt2 <br>
2. https://huggingface.co/gpt2?text=My+name+is+Merve+and+my+favorite <br>
3. https://www.philschmid.de/fine-tune-a-non-english-gpt-2-model-with-huggingface <br>
4. https://medium.com/swlh/everything-gpt-2-2-architecture-comprehensive-57129fac417a <br>
5. https://jalammar.github.io/illustrated-gpt2/ <br>
6. https://huggingface.co/docs/transformers/tasks/language_modeling <br>
7. https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html <br>
8. https://www.cerebras.net/machine-learning/to-bfloat-or-not-to-bfloat-that-is-the-question/#:~:text=bfloat16%20is%20a%2016%2Dbit,deep%20learning%20applications%20in%20mind. <br>
9. https://github.com/NVIDIA/apex <br>
10. 


