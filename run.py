import json
import re
from typing import List, Tuple, TypedDict, cast
from transformers import HfArgumentParser, set_seed, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LlamaForCausalLM, LlamaTokenizerFast, BitsAndBytesConfig
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation.candidate_generator import _crop_past_key_values
from accelerate import Accelerator
from dataclasses import dataclass, field
from datasets import Dataset, fingerprint
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
import wandb
import anthropic
from together import Together
import os
import string

@dataclass
class ScriptArguments:
    wandb_project: str = 'multiple-acsent'
    debug: bool = False
    out_dir: str = 'results'
    
    api: str | None = None
    model_name: str = "meta-llama/Meta-Llama-3-8B"
    batch_size: int | None = None
    seed: int = 42
    device: str = 'cuda:0'

    max_new_tokens: int = 10
    pz_end: int = 800
    pz_start: int = 20
    pz_count: int = 5
    pz_dist: str = 'uniform'
    num_test_examples: int = 100

    input_dim: int = 2
    max_w: int = 1000
    max_x: int = 1000


class IntegerRegressionModel(torch.nn.Module):
    def __init__(self, config: ScriptArguments):
        super(IntegerRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(config.input_dim, 1)
        self.linear.weight.data = torch.randint(0, config.max_w, self.linear.weight.data.shape).float()

    def forward(self, x):
        return torch.round(self.linear(x))

def get_dataset_generator(num_examples, model, config: ScriptArguments):
    def get_example():
        x = torch.randint(0, config.max_x, (config.input_dim,)).float()
        with torch.no_grad():
            y = model(x).int().numpy()[0]
        return x.int().numpy(), y
    for i in range(num_examples):
        # set_seed(config.seed + i) # make sure the same sequrence of examples
        examples = []
        targets = []
        for j in range(config.pz_end + 1):
            x, y = get_example()
            x_str = ', '.join([str(x_i) for x_i in x])
            examples.append(f'\nx={x_str}; y=')
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

def get_pz_list(config: ScriptArguments):
    if config.pz_dist == 'uniform':
        pz_list = np.linspace(config.pz_start, config.pz_end, config.pz_count, dtype=int)
    elif config.pz_dist == 'log':
        pz_list = np.logspace(np.log2(float(config.pz_start)), np.log2(float(config.pz_end)), config.pz_count, dtype=int, base=2)
    else:
        raise ValueError('Invalid pz_dist')

    if len(set(pz_list)) != len(pz_list):
        pz_list = list(set(pz_list))
        pz_list.sort()
        print('pz_list contains duplicate values, removing duplicates')

    return pz_list

def prompt_model(model: LlamaForCausalLM, tokenizer: LlamaTokenizerFast, batch, pz_list: List[int], config: ScriptArguments):
    generation_config = GenerationConfig(max_new_tokens=config.max_new_tokens, num_return_sequences=1, return_dict_in_generate=True, eos_token_id=[198, 11], pad_token_id=tokenizer.pad_token_id)
    kv_cache = DynamicCache()
    answers = []
    for pz in pz_list:
        prompt = build_prompt(batch['examples'], batch['targets'], pz)
        inputs = tokenizer(prompt, return_tensors="pt", padding='longest').to(config.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, generation_config=generation_config, past_key_values=kv_cache)
            # outputs = model.generate(**inputs, generation_config=generation_config)
        output_strs = tokenizer.batch_decode(outputs.sequences[:, inputs.input_ids.shape[1]:-1], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        answers_bi = [int_or_nan(x) for x in output_strs]
        if np.nan in answers_bi:
            print('Cannot parse output: ', prompt, output_strs)
        answers.append(answers_bi)
        kv_cache = outputs.past_key_values
        kv_cache = _crop_past_key_values(model, kv_cache, inputs.input_ids.shape[1])
    answers = torch.tensor(answers).T # (1, prompt_size)

    return answers

def prompt_claude(batch, pz_list, config):
    answers = []
    client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
    total_tokens = 0
    tqdm_obj = tqdm(pz_list)
    for pz in tqdm_obj:
        prompt = build_prompt(batch['examples'], batch['targets'], pz)[0]
        message = client.messages.create(
            model=config.model_name,
            max_tokens=10,
            temperature=0.0,
            system="Find the best fit given the existing datapoints. Respond with only an integer valued answer and nothing else. Do not include any explanations.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        answers_bi = int_or_nan(message.content[0].text)
        if answers_bi is np.nan:
            print('Cannot parse output: ', prompt, message.content[0].text)
        answers.append(answers_bi)
        total_tokens += message.usage.input_tokens
        tqdm_obj.set_postfix({'Total tokens sent': total_tokens})
    answers = torch.tensor(answers).unsqueeze(0) # (1, prompt_size)
    print('Total tokens sent: ', total_tokens)
    return answers

def prompt_together_ai(batch, pz_list, config):
    answers = []
    together = Together(api_key=os.environ['TOGETHER_AI_API_KEY'])

    total_tokens = 0
    tqdm_obj = tqdm(pz_list)
    
    for pz in tqdm_obj:
        prompt = build_prompt(batch['examples'], batch['targets'], pz)[0]
        
        output = together.completions.create(
            prompt=f"{prompt}",
            model=config.model_name,
            max_tokens=5,
            temperature=0.0,
            top_k=1,
            top_p=1
        )
        
        generated_text = output.choices[0].text
        total_tokens += output.usage.total_tokens

        answers_bi = int_or_nan(generated_text)
        if answers_bi is np.nan:
            print('Cannot parse output: ', prompt, generated_text)
        answers.append(answers_bi)

        tqdm_obj.set_postfix({'Total tokens used': total_tokens})

    answers = torch.tensor(answers).unsqueeze(0)  # (1, prompt_size)
    return answers

def evaluate(outputs, targets, config: ScriptArguments):
    acc = torch.mean((outputs == targets).float(), dim=0)

    outputs = outputs.float()
    targets = targets.float() # would casting first lose precision? 
    mse = torch.sqrt(torch.mean((outputs - targets) ** 2, dim=0))
    mse /= config.max_w * config.max_x # normalize the mse
    # breakpoint()
    return mse, acc

def int_or_nan(x: str):
    match = re.match("\d+", x.strip())
    if match:
        return int(match.group())
    else:
        return np.nan

def get_bnb_config(model_name):
    if 'meta-llama/Meta-Llama-3-70B' in model_name:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        return None

@torch.no_grad()
def main():
    parser = HfArgumentParser((ScriptArguments))

    config = cast(ScriptArguments, parser.parse_args_into_dataclasses()[0])

    set_seed(config.seed)
    hasher = fingerprint.Hasher()
    fp = hasher.hash(config)

    sim_model = IntegerRegressionModel(config)

    dataset = Dataset.from_generator(get_dataset_generator, gen_kwargs={'num_examples': config.num_test_examples, 'model': sim_model, 'config': config})
    dataset.set_format(type='torch', output_all_columns=True)

    if config.api is not None:
        model = None
        tokenizer = None
    else:
        bnb_config = get_bnb_config(config.model_name)
        model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=torch.bfloat16, device_map=config.device, quantization_config=bnb_config, attn_implementation="flash_attention_2")
        tokenizer = AutoTokenizer.from_pretrained(config.model_name, padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token

    wandb.init(project=config.wandb_project, config=config, mode='disabled', name=f'{config.model_name}-{config.input_dim}-{config.max_x}-{config.max_w}')

    pz_list = get_pz_list(config)
    if not os.path.exists(f'{config.out_dir}/cache/{fp}'):
        os.makedirs(f'{config.out_dir}/cache/{fp}')
        # calculate the batch size
        # bz = max(1, int(200 / pz)) if config.batch_size is None else config.batch_size
        bz = 1
        mse = torch.zeros(config.pz_count, dtype=torch.float32)
        tqdm_obj = tqdm(range(0, len(dataset), bz))
        all_answers = []
        all_targets = []
        for bi in tqdm_obj:
            batch = dataset[bi:bi+bz]
            if config.api == 'anthropic':
                assert bz == 1
                answers = prompt_claude(batch, pz_list, config)
                # print(answers)
            elif config.api == 'togetherai':
                assert bz == 1
                answers = prompt_together_ai(batch, pz_list, config)
            else:
                answers = prompt_model(model, tokenizer, batch, pz_list, config)
            targets = dataset['targets'][bi:bi+bz][:, pz_list]
            all_answers.append(answers)
            all_targets.append(targets)
        all_answers = torch.cat(all_answers, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        torch.save(all_answers, f'{config.out_dir}/cache/{fp}/answers.pt')
        torch.save(all_targets, f'{config.out_dir}/cache/{fp}/targets.pt')
        json.dump(config.__dict__, open(f'{config.out_dir}/cache/{fp}/config.json', 'w'))
    else:
        all_answers = torch.load(f'{config.out_dir}/cache/{fp}/answers.pt')
        all_targets = torch.load(f'{config.out_dir}/cache/{fp}/targets.pt')
        new_fp = hasher.hash(config.__dict__)
        os.makedirs(f'{config.out_dir}/cache/{new_fp}')
        torch.save(all_answers, f'{config.out_dir}/cache/{new_fp}/answers.pt')
        torch.save(all_targets, f'{config.out_dir}/cache/{new_fp}/targets.pt')
        json.dump(config.__dict__, open(f'{config.out_dir}/cache/{fp}/config.json', 'w'))
        # config = ScriptArguments(**json.load(open(f'{config.out_dir}/cache/{fp}/config.json')))
    mse, acc = evaluate(all_answers, all_targets, config)
    mse_table = wandb.Table(columns=['Prompt size', 'MSE'], data=[[pz_list[i], mse[i].item()] for i in range(len(mse))])
    wandb.log({'MSE vs Prompt Size': mse_table})
    acc_table = wandb.Table(columns=['Prompt size', 'Accuracy'], data=[[pz_list[i], acc[i].item()] for i in range(len(acc))])
    wandb.log({'Accuracy vs Prompt Size': acc_table})

if __name__ == '__main__':
    main()
