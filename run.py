from typing import List, TypedDict
from transformers import HfArgumentParser, set_seed, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LlamaForCausalLM, LlamaTokenizerFast
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation.candidate_generator import _crop_past_key_values
from accelerate import Accelerator
from dataclasses import dataclass, field
from datasets import Dataset
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
import wandb

@dataclass
class ScriptArguments:
    wandb_project: str = 'multiple-acsent'
    debug: bool = False
    
    model_name: str = "meta-llama/Meta-Llama-3-8B"
    batch_size: int | None = None
    seed: int = 42
    device: str = 'cuda:0'

    max_new_tokens: int = 10
    prompt_size: int = 800
    prompt_size_step: int = 20
    num_test_examples: int = 100
    out_file: str = 'results.csv'

    input_dim: int = 2
    max_w: int = 1000
    max_x: int = 1000


class IntegerRegressionModel(torch.nn.Module):
    def __init__(self, config: ScriptArguments):
        super(IntegerRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(config.input_dim, 1)
        self.linear.weight.data = torch.randint(0, 100, self.linear.weight.data.shape).float()

    def forward(self, x):
        return torch.round(self.linear(x))

def get_dataset_generator(num_examples, model, config: ScriptArguments):
    def get_example():
        x = torch.randint(0, config.max_x, (config.input_dim,)).float()
        with torch.no_grad():
            y = model(x).int().numpy()[0]
        return x.int().numpy(), y
    for i in range(num_examples):
        set_seed(config.seed + i) # make sure the same sequrence of examples
        examples = []
        targets = []
        for j in range(config.prompt_size + 1):
            x, y = get_example()
            examples.append(f'\nx_{j+1}={str(x)[1:-1]}, y=')
            targets.append(y)
        yield {
            'examples': examples,
            'targets': targets
        }

def build_prompt(examples: List[List[str]], targets: List[List[int]], pz: int):
    # only return the diff between the last prompt size and the current prompt size
    prompt = []
    for j in range(len(examples)):
        prompt.append(
            ''.join([f'{examples[j][i]}{targets[j][i]}' for i in range(pz)]) + examples[j][pz]
        )
    
    return prompt   

def prompt_model(model: LlamaForCausalLM, tokenizer: LlamaTokenizerFast, batch, config: ScriptArguments):
    generation_config = GenerationConfig(max_new_tokens=config.max_new_tokens, num_return_sequences=1, return_dict_in_generate=True, eos_token_id=[198, 11], pad_token_id=tokenizer.pad_token_id)

    kv_cache = DynamicCache()
    answers = []
    for pz in range(config.prompt_size_step, config.prompt_size + 1, config.prompt_size_step):
        prompt = build_prompt(batch['examples'], batch['targets'], pz)
        inputs = tokenizer(prompt, return_tensors="pt", padding='longest').to(config.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, generation_config=generation_config, past_key_values=kv_cache)
            # outputs = model.generate(**inputs, generation_config=generation_config)
        output_strs = tokenizer.batch_decode(outputs.sequences[:, inputs.input_ids.shape[1]:-1], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        answers_bi = [int_or_nan(x) for x in output_strs]
        assert np.nan not in answers_bi, breakpoint()
        answers.append(answers_bi)
        kv_cache = outputs.past_key_values
        kv_cache = _crop_past_key_values(model, kv_cache, inputs.input_ids.shape[1])
    answers = torch.tensor(answers).T # (1, prompt_size)

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

    sim_model = IntegerRegressionModel(config)

    dataset = Dataset.from_generator(get_dataset_generator, gen_kwargs={'num_examples': config.num_test_examples, 'model': sim_model, 'config': config})
    dataset.set_format(type='torch', output_all_columns=True)

    model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=torch.bfloat16, device_map=config.device)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    
    wandb.init(project=config.wandb_project, config=config, magic=True, mode='disabled' if config.debug else 'online')

    # calculate the batch size
    # bz = max(1, int(200 / pz)) if config.batch_size is None else config.batch_size
    bz = 1
    mse = torch.zeros(config.prompt_size // config.prompt_size_step, dtype=torch.float32)
    for bi in tqdm(range(0, len(dataset), bz)):
        batch = dataset[bi:bi+bz]
        answers = prompt_model(model, tokenizer, batch, config)
        targets = dataset['targets'][bi:bi+bz, config.prompt_size_step::config.prompt_size_step]
        mse += evaluate(answers, targets)
    mse /= len(dataset)
    result_table = wandb.Table(columns=['Prompt size', 'MSE'], data=[[(i+1)*config.prompt_size_step, mse[i].item()] for i in range(len(mse))])
    wandb.log({'MSE vs Prompt Size': result_table})

if __name__ == '__main__':
    main()
