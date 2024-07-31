import json
import re
from typing import List, Literal, Optional, Tuple, TypedDict, cast
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
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
from random import sample

@dataclass
class MetaArguments:
    use_cache: bool = True

@dataclass
class ScriptArguments:
    # project arguments
    wandb_project: str = 'multiple-acsent'
    debug: bool = False
    out_dir: str = 'results'
    seed: int = 42
    device: str = 'cuda:0'

    # model arguments
    api: str | None = None
    model_name: str = "meta-llama/Meta-Llama-3-8B"
    batch_size: int | None = 1
    max_new_tokens: int = 10

    # dataset arguments
    pz_end: int = 800
    pz_start: int = 20
    pz_count: int = 5
    pz_dist: str = 'uniform'
    num_test_examples: int = 100
    test_x_range: Optional[List[int]] = None

    input_dim: int = 2
    w_range: List[int] = field(default_factory=lambda: [0, 1000])
    x_range: List[int] = field(default_factory=lambda: [0, 1000])
    dataset_type: Literal['default', 'shuffle', 'perturb-input', 'perturb-context'] = 'default'
    
    same_last_example: bool = True
    
    def __post_init__(self):
        if self.test_x_range is None:
            self.test_x_range = self.x_range

class IntegerRegressionModel(torch.nn.Module):
    def __init__(self, config: ScriptArguments):
        super(IntegerRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(config.input_dim, 1, bias=False)
        self.linear.weight.data = torch.randint(*config.w_range, self.linear.weight.data.shape).float()

    def forward(self, x):
        return torch.round(self.linear(x))

def get_dataset_generator(num_examples, model, pz_list, config: ScriptArguments):
    def get_example(r):
        x = torch.randint(*r, (config.input_dim,)).float()
        with torch.no_grad():
            y = model(x).int()[0]
        return x.int(), y

    last_example = get_example(config.test_x_range)
    for i in range(num_examples):
        if i > 0 and config.dataset_type == 'shuffle':
            def shuffle(x):
                # shuffle the context except the last example
                return x[torch.randperm(len(x) - 1).tolist() + [len(x)-1]]
            xs, ys = [shuffle(xp) for xp in xs], [shuffle(yp) for yp in ys]
        elif i > 0 and config.dataset_type == 'perturb-input':
            # resample the last example (test_x_range is small) and set it for every prompt size
            # keep everything else the same
            last_example = get_example(config.test_x_range)
            for xp in xs:
                xp[-1] = last_example[0]
            for yp in ys:
                yp[-1] = last_example[1]
        else:
            # Generate fresh examples for each prompt size
            xs = []
            ys = []
            for pz in pz_list:
                xp, yp = [], []
                for j in range(pz + 1):
                    if j == pz:
                        if config.same_last_example:
                            x, y = last_example
                        else:
                            x, y = get_example(config.test_x_range)
                    else:
                        x, y = get_example(config.x_range)
                    xp.append(x)
                    yp.append(y)
                xs.append(torch.stack(xp))
                ys.append(torch.stack(yp))
        yield {
            'xs': xs,
            'ys': ys
        }

def build_prompt(batch, pzi: int, text_format=True):
    xp = batch['xs'][0][pzi].numpy()
    yp = batch['ys'][0][pzi].numpy()
    if text_format:
        prompt = ''
        target = yp[-1].item() # no formatting needed
        for i, (x, y) in enumerate(zip(xp, yp)):
            x_str = '[' + ', '.join(map(str, x)) + ']'
            if i < len(xp) - 1:
                prompt += f'\nx={x_str}; y={y}'
            else:
                prompt += f'\nx={x_str}; y='
        return prompt, target
    else:
        x_train = xp[:-1]
        y_train = yp[:-1]
        x_test = xp[-1:]
        y_test = yp[-1:].item()
        return x_train, y_train, x_test, y_test

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
    targets = []
    for pzi, pz in enumerate(pz_list):
        prompt, target = build_prompt(batch, pzi)
        inputs = tokenizer(prompt, return_tensors="pt", padding='longest').to(config.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, generation_config=generation_config, past_key_values=kv_cache)
            # outputs = model.generate(**inputs, generation_config=generation_config)
        output_strs = tokenizer.batch_decode(outputs.sequences[:, inputs.input_ids.shape[1]:-1], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        answers_bi = [int_or_nan(x) for x in output_strs]
        if np.nan in answers_bi:
            print('Cannot parse output: ', prompt, output_strs)
        answers.append(answers_bi)
        targets.append(target)
        kv_cache = outputs.past_key_values
        kv_cache = _crop_past_key_values(model, kv_cache, inputs.input_ids.shape[1])
    answers = torch.tensor(answers).T # (1, prompt_size)
    targets = torch.tensor(targets).unsqueeze(0) # (1, prompt_size)

    return answers, targets

def prompt_claude(batch, pz_list, config):
    answers = []
    targets = []
    client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
    total_tokens = 0
    tqdm_obj = tqdm(pz_list)
    for pzi, pz in enumerate(tqdm_obj):
        prompt, target = build_prompt(batch, pzi)
        message = client.messages.create(
            model=config.model_name,
            max_tokens=config.max_new_tokens,
            temperature=0.0,
            system="Your only job is to find the answer given the provided datapoints. Respond with only an integer valued answer and nothing else. Do not include any explanations. Do not output any steps.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        answers_bi = int_or_nan(message.content[0].text)
        if answers_bi is np.nan:
            print('Cannot parse output: ', prompt, message.content[0].text)
        answers.append(answers_bi)
        targets.append(target)
        total_tokens += message.usage.input_tokens
        tqdm_obj.set_postfix({'Sent': message.usage.input_tokens})
    answers = torch.tensor(answers).unsqueeze(0) # (1, prompt_size)
    targets = torch.tensor(targets).unsqueeze(0) # (1, prompt_size)
    print('Total tokens sent: ', total_tokens)
    return answers, targets

def prompt_together_ai(batch, pz_list, config: ScriptArguments):
    is_chat = 'instruct' in config.model_name.lower()
    
    answers = []
    targets = []
    together = Together(api_key=os.environ['TOGETHER_AI_API_KEY'])

    total_tokens = 0
    tqdm_obj = tqdm(pz_list)
    
    for pzi, pz in enumerate(tqdm_obj):
        prompt, target = build_prompt(batch, pzi)
        if is_chat:
            output = together.chat.completions.create(
                messages=[
                    {"role": "system", "content": 'Your only job is to find the answer given the provided datapoints. Respond with only an integer valued answer and nothing else. Do not include any explanations.'},
                    {"role": "user", "content": prompt},
                ],
                model=config.model_name,
                max_tokens=config.max_new_tokens,
                temperature=0.0,
                top_k=1,
                top_p=1
            )
            generated_text = output.choices[0].message.content
        else:
            output = together.completions.create(
                prompt=f"{prompt}",
                model=config.model_name,
                max_tokens=config.max_new_tokens,
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
        targets.append(target)

        tqdm_obj.set_postfix({'Sent': output.usage.total_tokens})
    print('Total tokens sent: ', total_tokens)

    answers = torch.tensor(answers).unsqueeze(0)  # (1, prompt_size)
    targets = torch.tensor(targets).unsqueeze(0)  # (1, prompt_size)
    return answers, targets

def get_baseline(batch, pz_list, config: ScriptArguments):
    answers = []
    targets = []
    for pzi, pz in enumerate(pz_list):
        x_train, y_train, x_test, y_test = build_prompt(batch, pzi, text_format=False)
        if 'KNN' in config.model_name:
            if '-' in config.model_name:
                k = int(config.model_name.split('-')[1])
            else:
                k = np.round(np.sqrt(pz)).astype(int)
            model = KNeighborsRegressor(n_neighbors=k)
        elif config.model_name == 'Linear':
            model = LinearRegression()
        else:
            raise ValueError('Invalid model_name')
        answers_bi = model.fit(x_train, y_train).predict(x_test).tolist()
        answers.append(answers_bi)
        targets.append(y_test)
    answers = torch.tensor(answers).T # (1, prompt_size)
    targets = torch.tensor(targets).unsqueeze(0) # (1, prompt_size)

    return answers, targets

def evaluate(outputs, targets, config: ScriptArguments):
    acc = torch.mean((outputs == targets).float(), dim=0)

    outputs = outputs.float()
    targets = targets.float() # would casting first lose precision? 
    mse = torch.sqrt(torch.mean((outputs - targets) ** 2, dim=0))
    mse /= (config.w_range[1] - config.w_range[0]) * (config.x_range[1] - config.x_range[0]) # normalize the mse
    # breakpoint()
    return mse, acc

def int_or_nan(x: str):
    match = re.match("\d+", x.strip())
    if match:
        num = int(match.group())
        try:
            torch.tensor(num, dtype=torch.long)
            return int(num)
        except:
            return np.nan
        
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
    config, meta_args = HfArgumentParser((ScriptArguments, MetaArguments)).parse_args_into_dataclasses()

    config = cast(ScriptArguments, config)
    meta_args = cast(MetaArguments, meta_args) # meta args are not used for fingerprinting

    set_seed(config.seed)
    # fp = repr(config)
    fp = fingerprint.Hasher().hash(tuple(sorted(config.__dict__.items())))
    print(config)
    print(fp)

    cache_found = False
    if meta_args.use_cache and os.path.exists(f'{config.out_dir}/cache/{fp}'):
        try:
            all_answers = torch.load(f'{config.out_dir}/cache/{fp}/answers.pt')
            all_targets = torch.load(f'{config.out_dir}/cache/{fp}/targets.pt')
            cache_found = True
        except:
            print('Cache not found')

    sim_model = IntegerRegressionModel(config)
    print(sim_model.state_dict())

    pz_list = get_pz_list(config)
    dataset = Dataset.from_generator(get_dataset_generator, gen_kwargs={'num_examples': config.num_test_examples, 'model': sim_model, 'pz_list': pz_list, 'config': config})
    dataset.set_format(type='torch', output_all_columns=True)

    if not cache_found:
        os.makedirs(f'{config.out_dir}/cache/{fp}', exist_ok=True)

        if config.api is not None:
            model = None
            tokenizer = None
        else:
            bnb_config = get_bnb_config(config.model_name)
            model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=torch.bfloat16, device_map=config.device, quantization_config=bnb_config, attn_implementation="flash_attention_2")
            tokenizer = AutoTokenizer.from_pretrained(config.model_name, padding_side='left')
            tokenizer.pad_token = tokenizer.eos_token

        wandb.init(project=config.wandb_project, config=config, mode='disabled', name=f'{config.model_name}-{config.input_dim}-{config.x_range}-{config.w_range}')

        # calculate the batch size
        # bz = max(1, int(200 / pz)) if config.batch_size is None else config.batch_size
        bz = config.batch_size
        assert bz == 1, 'Batch size must be 1'
        mse = torch.zeros(config.pz_count, dtype=torch.float32)
        tqdm_obj = tqdm(range(0, len(dataset), bz))
        all_answers = []
        all_targets = []
        for bi in tqdm_obj:
            batch = dataset[bi:bi+bz]
            retries = 3
            while retries > 0:
                try:
                    if config.api == 'anthropic':
                        answers, targets = prompt_claude(batch, pz_list, config)
                        # print(answers)
                    elif config.api == 'togetherai':
                        answers, targets = prompt_together_ai(batch, pz_list, config)
                    elif config.api == 'baseline':
                        answers, targets = get_baseline(batch, pz_list, config)
                    elif config.api is None:
                        answers, targets = prompt_model(model, tokenizer, batch, pz_list, config)
                    else:
                        raise ValueError('Invalid api value')
                    break
                except Exception as e:
                    print('Error: ', e)
                    retries -= 1
                    continue
            all_answers.append(answers)
            all_targets.append(targets)
        all_answers = torch.cat(all_answers, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        torch.save(all_answers, f'{config.out_dir}/cache/{fp}/answers.pt')
        torch.save(all_targets, f'{config.out_dir}/cache/{fp}/targets.pt')
        dataset.to_csv(f'{config.out_dir}/cache/{fp}/dataset.csv')
        json.dump(config.__dict__, open(f'{config.out_dir}/cache/{fp}/config.json', 'w'))
        torch.save(sim_model.state_dict(), f'{config.out_dir}/cache/{fp}/model.pt')
        print('Saved cache to ', f'{config.out_dir}/cache/{fp}')

        # Not using wandb for now
        # mse, acc = evaluate(all_answers, all_targets, config)
        # mse_table = wandb.Table(columns=['Prompt size', 'MSE'], data=[[pz_list[i], mse[i].item()] for i in range(len(mse))])
        # wandb.log({'MSE vs Prompt Size': mse_table})
        # acc_table = wandb.Table(columns=['Prompt size', 'Accuracy'], data=[[pz_list[i], acc[i].item()] for i in range(len(acc))])
        # wandb.log({'Accuracy vs Prompt Size': acc_table})
    else:
        all_answers = torch.load(f'{config.out_dir}/cache/{fp}/answers.pt')
        all_targets = torch.load(f'{config.out_dir}/cache/{fp}/targets.pt')

if __name__ == '__main__':
    main()
