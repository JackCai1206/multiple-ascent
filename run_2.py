from typing import List, TypedDict
from transformers import HfArgumentParser, set_seed, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LlamaForCausalLM, LlamaTokenizerFast
from transformers.cache_utils import Cache, DynamicCache
from accelerate import Accelerator
from dataclasses import dataclass, field
from datasets import Dataset
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
import wandb
import os
from threading import Thread

@dataclass
class ScriptArguments:
    model_name: str = "meta-llama/Meta-Llama-3-8B"
    batch_size: int | None = None
    seed: int = 42
    input_dim: int = 2
    device: str = 'cuda:0'
    max_new_tokens: int = 1
    prompt_size: int = 10
    num_test_examples: int = 100
    out_file: str = 'results.csv'
    wandb_project: str = 'multiple-acsent'
    debug: bool = False
    kv_cache_path: str = 'temp'

class IntegerRegressionModel(torch.nn.Module):
    def __init__(self, config):
        super(IntegerRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(config.input_dim, 1)
        self.linear.weight.data = torch.randint(0, 10, self.linear.weight.data.shape).float()

    def forward(self, x):
        return torch.round(self.linear(x))

def get_dataset_generator(num_examples, model, config):
    def get_example():
        x = torch.randint(0, 10, (config.input_dim,)).float()
        with torch.no_grad():
            y = model(x).int().numpy()[0]
        return x.int().numpy(), y
    for i in range(num_examples):
        set_seed(config.seed + i) # make sure the same sequrence of examples
        examples = []
        targets = []
        for j in range(config.prompt_size + 1):
            x, y = get_example()
            examples.append(f'\nx_{j+1}={str(x)[1:-1]},y=')
            targets.append(y)
        yield {
            'examples': examples,
            'targets': targets
        }

def build_prompt(examples: List[List[str]], targets: List[List[int]], prompt_size: int):
    return [
        ''.join([f'{examples[j][i]}{targets[j][i]}' for i in range(prompt_size)]) + examples[j][prompt_size]
        for j in range(len(examples))
    ]

# def slice_kv_cache(kv_cache: DynamicCache | None, bi, bz) -> DynamicCache:
#     new_cache = DynamicCache()
#     if kv_cache is None:
#         return new_cache
#     for layer, (keys, values) in enumerate(kv_cache):
#         new_cache.update(keys[bi:bi+bz], values[bi:bi+bz], layer)
#     return new_cache

# def cat_kv_cache(kv_cache_list: List[DynamicCache]):
#     new_cache = DynamicCache()
#     for layer in range(len(kv_cache_list[0])):
#         keys = torch.cat([kv_cache.key_cache[layer] for kv_cache in kv_cache_list], dim=0)
#         values = torch.cat([kv_cache.value_cache[layer] for kv_cache in kv_cache_list], dim=0)
#         new_cache.update(keys, values, layer)
#     return new_cache

# def move_cache_to_device(kv_cache: DynamicCache, device):
#     kv_cache.key_cache = [keys.to(device) for keys in kv_cache.key_cache]
#     kv_cache.value_cache = [values.to(device) for values in kv_cache.value_cache]
#     return kv_cache

# def save_cache(kv_cache: DynamicCache, path, bi):
#     cache_bz = kv_cache.key_cache[0].shape[0]
#     for i in range(cache_bz):
#         single_bz_cache = slice_kv_cache(kv_cache, i, 1)
#         torch.save(single_bz_cache, os.path.join(path, f'cache_{bi+i}.pt'))

# def load_cache(path, bi, bz):
#     caches = []
#     for i in range(bz):
#         try:
#             single_bz_cache = torch.load(os.path.join(path, f'cache_{bi + i}.pt'))
#         except:
#             return DynamicCache()
#         caches.append(single_bz_cache)
#     return cat_kv_cache(caches)
    

def prompt_model(model: LlamaForCausalLM, tokenizer: LlamaTokenizerFast, dataset, config: ScriptArguments, pz=1):
    generation_config = GenerationConfig(max_new_tokens=config.max_new_tokens, num_return_sequences=1, return_dict_in_generate=True)

    # calculate the batch size
    bz = max(1, int(200 / pz)) if config.batch_size is None else config.batch_size
    answers = []
    for bi in range(0, len(dataset), bz):
        batch = dataset[bi:bi+bz]
        prompt = build_prompt(batch['examples'], batch['targets'], pz)
        inputs = tokenizer(prompt, return_tensors="pt", padding='longest').to(config.device)
        with torch.no_grad():
            # kv_cache_slice = load_cache(config.kv_cache_path, bi, bz)
            # move_cache_to_device(kv_cache_slice, config.device)
            # outputs = model.generate(**inputs, generation_config=generation_config, past_key_values=kv_cache_slice)
            # del kv_cache_slice
            outputs = model.generate(**inputs, generation_config=generation_config)
        answers_bi = tokenizer.batch_decode(outputs.sequences[:, inputs.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        answers_bi = [int_or_nan(x) for x in answers_bi]
        answers.extend(answers_bi)
        # kv_cache_slice = move_cache_to_device(outputs.past_key_values, 'cpu')
        # Thread(target=lambda: save_cache(kv_cache_slice, config.kv_cache_path, bi)).start()
    answers = torch.tensor(answers) # len(dataset)

    return answers

def evaluate(outputs, targets):
    return torch.sum((outputs - targets).float() ** 2, dim=0)

def int_or_nan(x):
    try:
        return int(x)
    except:
        return np.nan

@torch.no_grad()
def main():
    parser = HfArgumentParser((ScriptArguments))

    config: ScriptArguments = parser.parse_args_into_dataclasses()[0]

    set_seed(config.seed)
    
    if os.path.exists(config.kv_cache_path):
        os.system(f'rm -rf {config.kv_cache_path}')
    os.makedirs(config.kv_cache_path, exist_ok=True)

    sim_model = IntegerRegressionModel(config)

    dataset = Dataset.from_generator(get_dataset_generator, gen_kwargs={'num_examples': config.num_test_examples, 'model': sim_model, 'config': config})
    dataset.set_format(type='torch', output_all_columns=True)

    model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=torch.bfloat16, device_map=config.device)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    
    wandb.init(project=config.wandb_project, config=config, magic=True, mode='disabled' if config.debug else 'online')

    answers = []
    for pz in tqdm(range(20, config.prompt_size + 1, 20), desc='Prompt size'):
        mse = 0
        answers = prompt_model(model, tokenizer, dataset, config, pz)
        targets = dataset['targets'][:, pz] # We don't prompt with 0 examples, so skip the first target
        mse += evaluate(answers, targets)
        mse /= len(dataset)
        metrics = {'prompt_size': pz, 'mse': mse.item()}
        wandb.log(metrics)

if __name__ == '__main__':
    main()
