# Baselines Training
Here we provide instructions on how to train your own baselines. Firstly, you need to git pull the data from [HF](https://huggingface.co/MLLMMU/baseline_train_split/tree/main) to your local folder. 

## GA

```python
python GA.py \
	--model_id llava-hf/llava-1.5-7b-hf \
	--vanilla_dir [Vanilla Model Path] \
	--data_split_dir [Your local data Path that you downloaded from [HF](https://huggingface.co/MLLMMU/baseline_train_split/tree/main)] \
	--forget_split_ratio 5 \
	--save_dir [GA Saved Path] \
	--batch_size 4 \
	--lr 2e-5 \
	--num_epochs 1 \
	--max_length 384
```

## GA Difference
Since the script can locate forget/retain automatically, we only need to pass in one parameter to identify the data split directory.
```python
python GA_Difference.py \
	--model_id llava-hf/llava-1.5-7b-hf \
	--vanilla_dir [Vanilla Model Path] \
	--data_split_dir [Your local data Path that you downloaded from [HF](https://huggingface.co/MLLMMU/baseline_train_split/tree/main)] \
	--forget_split_ratio 5 \
	--save_dir [GA Difference Saved Path] \
	--batch_size 4 \
	--lr 2e-5 \
	--num_epochs 1 \
	--max_length 384
```

## KL Minimization
```python
python KL_Min.py \
	--model_id llava-hf/llava-1.5-7b-hf \
	--vanilla_dir [Vanilla Model Path] \
	--data_split_dir [Your local data Path that you downloaded from [HF](https://huggingface.co/MLLMMU/baseline_train_split/tree/main)] \
	--forget_split_ratio 5 \
	--save_dir [KL Min Saved Path] \
	--batch_size 4 \
	--lr 2e-5 \
	--num_epochs 1 \
	--max_length 384
```

## NPO 
Since NPO demands a reference model where we call "oracle model", so you would need to train the reference model first. In our case, we used the retain dataset to finetune the original model (i.e. off-the-shelf model) and then regard it as the reference model. Hence, you would need to first run the following command to obtain the reference model:

```python
python reference_model_FT.py \
	--model_id llava-hf/llava-1.5-7b-hf \
	--data_split_dir [Your local data Path that you downloaded from [HF](https://huggingface.co/MLLMMU/baseline_train_split/tree/main)] \
	--forget_split_ratio 5 \
	--save_dir [oracle model Saved Path] \
	--batch_size 4 \
	--lr 2e-5 \
	--num_epochs 1 \
	--max_length 384
```
Then, using this reference model, you may run the NPO baseline:

```python
python NPO.py \
	--model_id llava-hf/llava-1.5-7b-hf \
	--vanilla_dir [Vanilla Model Path] \
	--oracle_model_id [oracle model saved path] \
	--data_split_dir [Your local data Path that you downloaded from [HF](https://huggingface.co/MLLMMU/baseline_train_split/tree/main)] \
	--forget_split_ratio 5 \
	--save_dir [NPO Saved Path] \
	--batch_size 4 \
	--lr 2e-5 \
	--num_epochs 1 \
	--max_length 384
```
