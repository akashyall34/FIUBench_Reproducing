import copy
import os
import sys
from collections import defaultdict, Counter

import pandas as pd
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from peft import PeftModel
sys.path.append(('../'))
sys.path.append(('../../'))
from datasets import load_dataset, Dataset
import random
import torch
import os
import json
from torch.utils.data import Subset
import argparse
from PIL import Image
import torch
from transformers import BitsAndBytesConfig, LlavaForConditionalGeneration, AutoProcessor, get_scheduler, AdamW, \
    LlavaNextForConditionalGeneration, LlavaNextProcessor, Idefics2ForConditionalGeneration, AutoTokenizer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import json
from data_process.data_preprocess import Vanilla_LLaVA_Dataset, train_collate_fn_llava, train_collate_fn, \
    train_collate_fn_idefics, LLAVA_multimodal_Dataset
import matplotlib.pyplot as plt
from PIL import Image
from accelerate import Accelerator
from transformers import AutoProcessor
from transformers import BitsAndBytesConfig, LlavaForConditionalGeneration
import torch
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import Trainer, TrainingArguments
# from trl import SFTConfig, SFTTrainer
import random
from torch.utils.data import Subset
from torch.nn import functional as F

def update_json_id(json_folder):
    # Loop through all files in the folder
    for filename in os.listdir(json_folder):
        if filename.endswith(".json"):
            json_path = os.path.join(json_folder, filename)

            # Load the JSON file
            with open(json_path, 'r') as file:
                data = json.load(file)

            # Get the global "ID"
            global_id = data.get("ID")

            # Update the "ID" field inside each metadata entry
            for entry in data.get("metadata", []):
                entry["ID"] = global_id  # Set the entry "ID" to match the global "ID"

            # Save the updated JSON back to the file
            with open(json_path, 'w') as file:
                json.dump(data, file, indent=4)

            print(f"Updated {filename}")
def find_profiles_with_details(json_dir, output_file):
    """
    Finds profiles with duplicate names and records their IDs, and counts the number of occurrences of jobs and born places.

    Args:
        json_dir (str): Directory containing the profile JSON files.
        output_file (str): Path to the output JSON file to save the results.
    """
    # Dictionary to hold names, employment types, birthplaces with associated IDs
    name_to_ids = defaultdict(list)
    employment_counter = Counter()
    born_place_counter = Counter()

    # Iterate over all JSON files in the directory
    for json_filename in os.listdir(json_dir):
        if json_filename.endswith(".json"):
            json_path = os.path.join(json_dir, json_filename)

            # Load the JSON data
            with open(json_path, 'r') as f:
                profile_data = json.load(f)

            # Extract relevant fields
            profile_id = profile_data.get("ID", "Unknown_ID")
            biography = profile_data.get("biography", {})

            name = biography.get("Name", "Unknown_Name")
            employment = biography.get("Employment", "Unknown_Employment")
            born_place = biography.get("Born", "Unknown_Born_Place")

            # Add the ID to the name's list (for duplicates check)
            name_to_ids[name].append(profile_id)

            # Count the number of occurrences of each job and born place
            employment_counter[employment] += 1
            born_place_counter[born_place] += 1

    # Only keep names that are duplicated (i.e., appear more than once)
    duplicated_names = {name: ids for name, ids in name_to_ids.items() if len(ids) > 1}

    # Prepare the final result to be saved
    results = {
        "duplicated_names": duplicated_names,  # Only save duplicated names and their IDs
        "employment_summary": employment_counter,  # Summary count of employment types
        "born_place_summary": born_place_counter  # Summary count of birth places
    }

    # Save the results to a JSON file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_file}")



def flatten_dataset(profiles):
    """
    Flatten the profiles dataset such that each question-answer pair becomes a single sample.
    Args:
        profiles (Subset or list): List of profiles or a subset of profiles.
    Returns:
        flattened_data (list): List of dictionaries with image path and each QA pair.
    """
    flattened_data = []
    for profile in profiles:
        # Each profile has an image path and metadata (question-answer pairs)
        image_path = profile["image_path"]
        # print(image_path)
        # print(profile)
        # Ensure the image is loaded correctly (do this in the flattening step)
        try:
            image = Image.open(image_path).convert("RGB")  # Load the image as PIL
        except Exception as e:
            print(f"Error loading image at {image_path}: {e}")
            continue  # Skip profiles with invalid images

        # Flatten the metadata (which contains the questions and answers)
        metadata = profile["metadata"]
        for qa_pair in metadata:
            question = qa_pair.get("Question", "")
            answer = qa_pair.get("Answer", "")

            flattened_data.append({
                "image": image,  # Include the image here
                "question": question,
                "answer": answer
            })

    return flattened_data


def split_and_save_dataset(dataset, save_dir, split_ratios):
    """
    Split the dataset based on profiles (not questions) and save the forget/retain sets, along with the selected IDs.
    Args:
        dataset: The Vanilla_LLaVA_Dataset_baseline, which contains profiles.
        save_dir: Directory to save the forget and retain datasets.
        split_ratios: List of percentages for splitting (e.g., [5, 10, 15]).
    """
    dataset_size = len(dataset)  # Number of profiles
    all_indices = list(range(dataset_size))  # Get indices for all profiles

    for forget_percentage in split_ratios:
        retain_percentage = 100 - forget_percentage

        forget_dir = os.path.join(save_dir, f"forget_{forget_percentage}")
        retain_dir = os.path.join(save_dir, f"retain_{retain_percentage}")

        forget_path = os.path.join(forget_dir, "forget_dataset.pt")
        retain_path = os.path.join(retain_dir, "retain_dataset.pt")
        forget_ids_path = os.path.join(forget_dir, "forget_ids.json")
        retain_ids_path = os.path.join(retain_dir, "retain_ids.json")

        if os.path.exists(forget_path) and os.path.exists(retain_path):
            print(f"Split for forget {forget_percentage}% and retain {retain_percentage}% already exists, skipping.")
            continue

        print(f"Creating split for forget {forget_percentage}% and retain {retain_percentage}%.")

        forget_size = int(forget_percentage / 100 * dataset_size)
        retain_size = dataset_size - forget_size

        # Randomize the indices to ensure a random split
        random.shuffle(all_indices)
        forget_indices = all_indices[:forget_size]
        retain_indices = all_indices[forget_size:forget_size + retain_size]

        forget_dataset = Subset(dataset, forget_indices)
        retain_dataset = Subset(dataset, retain_indices)

        # Debugging: Print metadata for the forget set
        print("Metadata for forget set:")
        for idx in forget_indices:
            print(dataset[idx]["metadata"])  # Print metadata of each selected profile for debugging
        print("\n")

        # Save the IDs of the selected forget and retain samples
        forget_ids = [dataset[idx]["ID"] for idx in forget_indices]  # Extract the ID field from each sample
        retain_ids = [dataset[idx]["ID"] for idx in retain_indices]  # Extract the ID field from each sample

        # Create the directories
        os.makedirs(forget_dir, exist_ok=True)
        os.makedirs(retain_dir, exist_ok=True)

        # Save datasets
        torch.save(forget_dataset, forget_path)
        torch.save(retain_dataset, retain_path)

        # Save the IDs in JSON format
        with open(forget_ids_path, 'w') as f:
            json.dump(forget_ids, f, indent=4)
        with open(retain_ids_path, 'w') as f:
            json.dump(retain_ids, f, indent=4)

        print(f"Saved forget {forget_percentage}% dataset and IDs to {forget_dir}")
        print(f"Saved retain {retain_percentage}% dataset and IDs to {retain_dir}")


def convert_to_hf_dataset(custom_dataset):
    data_dict = {
        "image": [],
        "question": [],
        "answer": []
    }

    # Iterate through your custom dataset and extract data
    for idx in range(len(custom_dataset)):
        sample = custom_dataset[idx]
        if sample:  # Check if sample is not empty
            data_dict["image"].append(sample["image"])  # Keep images as paths or tensors, whichever you prefer
            data_dict["question"].append(sample["question"])
            data_dict["answer"].append(sample["answer"])

    # Convert to Hugging Face Dataset
    hf_dataset = Dataset.from_dict(data_dict)

    return hf_dataset

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['multi_modal_projector', 'vision_model']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def visualize_batch(dataloader, processor, num_samples=3):
    # Get the first batch from the dataloader
    for batch in dataloader:
        input_ids = batch[0][:num_samples]  # Get first num_samples input_ids
        decoded_texts = processor.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        # Print the decoded text
        for i, text in enumerate(decoded_texts):
            print(f"\nSample {i + 1} - Decoded Text:")
            print(text)

        break  # Only visualize the first batch

def load_dataset_split(data_split_dir, split_ratio):
    """
    Load the corresponding forget and retain datasets based on the split_ratio argument.

    Args:
        data_split_dir (str): Base directory where the split datasets are stored.
        split_ratio (int): Percentage of forget set (e.g., 5, 10, 15).

    Returns:
        forget_dataset (torch.utils.data.Dataset): The forget dataset.
        retain_dataset (torch.utils.data.Dataset): The retain dataset.
    """
    # Define the corresponding folder names
    forget_dir = os.path.join(data_split_dir, f"forget_{split_ratio}")
    retain_dir = os.path.join(data_split_dir, f"retain_{100 - split_ratio}")

    # Define the paths to the saved datasets
    forget_path = os.path.join(forget_dir, "forget_dataset.pt")
    retain_path = os.path.join(retain_dir, "retain_dataset.pt")

    # Check if the paths exist and load the datasets
    if os.path.exists(forget_path) and os.path.exists(retain_path):
        print(f"Loading forget {split_ratio}% dataset from {forget_path}")
        forget_dataset = torch.load(forget_path)

        print(f"Loading retain {100 - split_ratio}% dataset from {retain_path}")
        retain_dataset = torch.load(retain_path)
    else:
        raise FileNotFoundError(
            f"Could not find forget or retain dataset for split ratio {split_ratio} in {data_split_dir}")

    return forget_dataset, retain_dataset


# Example usage:
def load_model_and_processor(args):
    """
    Load the model and processor based on the provided model_id.
    Different models may require different loading methods, which are handled with conditional statements.
    """
    if args.model_id.startswith("llava"):
        # Load LLAVA model and processor
        print("Loading LLAVA model...")
        model = LlavaForConditionalGeneration.from_pretrained(
            args.vanilla_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=True,
            # quantization_config=bnb_config,
            # cache_dir="/afs/crc.nd.edu/group/dmsquare/vol1/zliu29/mllm_unlearn/model/llava-1.5-7b-hf",
        )
        processor = AutoProcessor.from_pretrained(args.model_id)

    elif args.model_id.startswith("HuggingFaceM4"):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        # Load LLAVA Next model and processor
        print("Loading idefics2 model...")
        model = Idefics2ForConditionalGeneration.from_pretrained(
            args.vanilla_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            # quantization_config=bnb_config,
            low_cpu_mem_usage=True,
            cache_dir="/afs/crc.nd.edu/group/dmsquare/vol1/zliu29/mllm_unlearn/model/idfics2-8b",
        )
        processor = AutoProcessor.from_pretrained(
            "HuggingFaceM4/idefics2-8b",
            do_image_splitting=False
        )
    else:
        raise ValueError("Model ID not recognized or not supported. Please provide a valid model ID.")


    # Additional processor configuration if necessary
    processor.tokenizer.padding_side = "right"  # Ensure right padding
    processor.tokenizer.add_tokens(["<image>", "<pad>"], special_tokens=True)

    return model, processor



######################### Accelerate Version #################################
def main(args):

    # Load model and processor
    print("Trainer Status is ", args.trainer)
    model, processor = load_model_and_processor(args)
    model_vanilla = copy.deepcopy(model)  # Make a copy of the original model
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # Resize token embeddings to match the tokenizer
    # model.resize_token_embeddings(len(processor.tokenizer))
    # model_vanilla.resize_token_embeddings(len(processor.tokenizer))

    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print("WARNING: Resizing the embedding matrix to match the tokenizer vocab size.")
        model.resize_token_embeddings(len(tokenizer))
        model_vanilla.resize_token_embeddings(len(processor.tokenizer))

    # LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        # target_modules=["q_proj", "v_proj"],
        target_modules=find_all_linear_names(model),
        init_lora_weights="gaussian",
    )

    print("getting peft model")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # model.add_adapter(lora_config)
    # model.enable_adapters()
    if isinstance(model, PeftModel):
        print("This is a PEFT model.")
    else:
        print("This is NOT a PEFT model.")

    # Dataset and Dataloader setup

    # dataset = Vanilla_LLaVA_Dataset_baseline(json_dir=profile_dir, image_dir=image_base_path, flatten=False)
    # print(f"Dataset size (profiles): {len(dataset)}")

    forget_folder = os.path.join(args.data_split_dir, f"forget_{args.forget_split_ratio}")
    retain_folder = os.path.join(args.data_split_dir, f"retain_{100 - args.forget_split_ratio}")
    print("Forget Folder: ", forget_folder)
    print("Retain Folder: ", retain_folder)

    # Define paths to the Parquet files for "forget" and "retain" datasets
    forget_parquet_file = os.path.join(forget_folder, f"train-00000-of-00001.parquet")
    retain_parquet_file = os.path.join(retain_folder, f"train-00000-of-00001.parquet")

    # Load DataLoader
    forget_df = pd.read_parquet(forget_parquet_file)
    retain_df = pd.read_parquet(retain_parquet_file)

    multimodal_forget_dataset = LLAVA_multimodal_Dataset(df=forget_df)
    multimodal_retain_dataset = LLAVA_multimodal_Dataset(df=retain_df)


    if args.model_id.startswith("llava"):
        # Prepare the training dataloaders
        train_dataloader_forget = DataLoader(
            multimodal_forget_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda x: train_collate_fn_llava(x, processor, args)
        )
        train_dataloader_retain = DataLoader(
            multimodal_retain_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda x: train_collate_fn_llava(x, processor, args)
        )
    elif args.model_id.startswith("HuggingFaceM4"):
        train_dataloader_forget = DataLoader(
            multimodal_forget_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda x: train_collate_fn_idefics(x, processor, args)
        )
        train_dataloader_retain = DataLoader(
            multimodal_retain_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda x: train_collate_fn_idefics(x, processor, args)
        )
    else:
        raise ValueError("Model ID not recognized or not supported. Please provide a valid model ID.")

    accelerator = Accelerator()
    if args.gradient_accumulation:
        print("Gradient accumulation enabled.")
        accumulation_steps = 4  # Adjust based on memory
        model.gradient_checkpointing_enable()
    else:
        print("Gradient accumulation disabled.")

    optimizer = AdamW(model.parameters(), lr=args.lr)

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader_forget) + len(train_dataloader_retain)) * args.num_epochs,
    )

    # Prepare model, optimizer, and scheduler with accelerator
    model, optimizer, train_dataloader_forget, train_dataloader_retain, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader_forget, train_dataloader_retain, lr_scheduler
    )

    # Training loop with gradient ascent
    for epoch in range(args.num_epochs):
        model.train()
        total_loss_forget = 0
        total_loss_retain = 0

        # Forget dataset (gradient ascent)
        progress_bar_forget = tqdm(train_dataloader_forget, desc=f"Epoch {epoch + 1} - Forget Dataset")
        for step, batch in enumerate(progress_bar_forget):
            input_ids, attention_mask, pixel_values, labels = batch
            if args.gradient_accumulation:
                with accelerator.accumulate(model):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                    pixel_values=pixel_values, labels=labels)
                    loss_forget = -(outputs.loss / accumulation_steps)  # Gradient ascent with accumulation

                    accelerator.backward(loss_forget)

                    if (step + 1) % accumulation_steps == 0:
                        accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                pixel_values=pixel_values, labels=labels)
                loss_forget = -outputs.loss  # Gradient ascent without accumulation

                accelerator.backward(loss_forget)

                optimizer.step()
                optimizer.zero_grad()

            total_loss_forget += loss_forget.item()
            progress_bar_forget.set_postfix(loss=total_loss_forget / (step + 1))

        # Retain dataset (KL minimization + gradient descent)
        progress_bar_retain = tqdm(train_dataloader_retain, desc=f"Epoch {epoch + 1} - Retain Dataset")
        for step, batch in enumerate(progress_bar_retain):
            input_ids, attention_mask, pixel_values, labels = batch
            if args.gradient_accumulation:
                with accelerator.accumulate(model):
                    outputs_current = model(input_ids=input_ids, attention_mask=attention_mask,
                                            pixel_values=pixel_values, labels=labels)

                    # Original model output (without unlearning)
                    with torch.no_grad():
                        outputs_original = model_vanilla(input_ids=input_ids, attention_mask=attention_mask,
                                                         pixel_values=pixel_values, labels=labels)

                    # KL divergence loss
                    kl_div = F.kl_div(
                        F.log_softmax(outputs_current.logits, dim=-1),
                        F.softmax(outputs_original.logits, dim=-1),
                        reduction='batchmean'
                    )

                    # Total loss = conventional loss + KL divergence
                    loss_retain = (outputs_current.loss + kl_div) / accumulation_steps  # Gradient descent with accumulation

                    accelerator.backward(loss_retain)

                    if (step + 1) % accumulation_steps == 0:
                        accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad()
            else:
                outputs_current = model(input_ids=input_ids, attention_mask=attention_mask,
                                        pixel_values=pixel_values, labels=labels)

                # Original model output (without unlearning)
                with torch.no_grad():
                    outputs_original = model_vanilla(input_ids=input_ids, attention_mask=attention_mask,
                                                     pixel_values=pixel_values, labels=labels)

                # KL divergence loss
                kl_div = F.kl_div(
                    F.log_softmax(outputs_current.logits, dim=-1),
                    F.softmax(outputs_original.logits, dim=-1),
                    reduction='batchmean'
                )

                # Total loss = conventional loss + KL divergence
                loss_retain = outputs_current.loss + kl_div  # Gradient descent without accumulation

                accelerator.backward(loss_retain)

                optimizer.step()
                optimizer.zero_grad()

            total_loss_retain += loss_retain.item()
            progress_bar_retain.set_postfix(loss=total_loss_retain / (step + 1))

        # Step the learning rate scheduler after each epoch
        lr_scheduler.step()

        print(f"Epoch {epoch + 1} - Forget Loss: {total_loss_forget / len(train_dataloader_forget)}")
        print(f"Epoch {epoch + 1} - Retain Loss: {total_loss_retain / len(train_dataloader_retain)}")

    # Save the final model after training
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model = unwrapped_model.merge_and_unload()
    unwrapped_model.save_pretrained(args.save_dir)
    print(f"Model saved to: {args.save_dir}")

if __name__ == "__main__":
    # Argument parser for different options
    parser = argparse.ArgumentParser(description="Fine-tune different models")
    parser.add_argument("--model_id", type=str, required=True, help="Pretrained model ID")
    parser.add_argument("--vanilla_dir", type=str, required=True, help="Pretrained model ID")
    parser.add_argument("--save_dir", type=str, default="./saved_model", help="Directory to save the model")
    parser.add_argument("--data_split_dir", type=str, default="../Data_split", help="Directory to save the model")
    parser.add_argument("--forget_split_ratio", type=int, default=5, help="Directory to save the model")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs for training")
    parser.add_argument("--max_length", type=int, default=384, help="Maximum sequence length")
    parser.add_argument("--gradient_accumulation", type=bool, default=False, help="Enable gradient accumulation")
    parser.add_argument("--trainer", type=bool, default=False, help="Use HuggingFace Trainer")

    args = parser.parse_args()

    # Call main function
    main(args)
