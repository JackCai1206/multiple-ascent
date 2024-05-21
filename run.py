from typing import List
from transformers import HfArgumentParser, set_seed, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LlamaForCausalLM, LlamaTokenizerFast
from accelerate import Accelerator
from dataclasses import dataclass, field
from datasets import Dataset
import torch
import pandas as pd

@dataclass
class ScriptArguments:
    model_name: str = "meta-llama/Meta-Llama-3-8B"
    batch_size: int = 32
    seed: int = 42
    input_dim: int = 2
    device: str = 'cuda:0'
    max_new_tokens: int = 1
    prompt_size: int = 10
    num_test_examples: int = 1000
    out_file: str = 'results.csv'

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
        y = model(x).int().numpy()[0]
        return x.int().numpy(), y
    for i in range(num_examples):
        line = ''
        for j in range(config.prompt_size):
            x, y = get_example()
            line += f'\nx_{j}={str(x)[1:-1]},y={y}'
        last_x, last_y = get_example()
        line += f'\nx_{j+1}={str(last_x)[1:-1]},y='
        yield {
            'prompt': line,
            'answer': last_y
        }

def prompt_model(model: LlamaForCausalLM, tokenizer: LlamaTokenizerFast, prompt: List[str], config: ScriptArguments):
    generation_config = GenerationConfig(max_new_tokens=config.max_new_tokens, num_return_sequences=1)
    inputs = tokenizer(prompt, return_tensors="pt", padding='longest').to(config.device)
    outputs = model.generate(**inputs, generation_config=generation_config)
    return tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=True)

def evaluate(outputs, targets):
    return torch.mean((outputs - targets).float() ** 2)

def log_metrics(metrics, df: pd.DataFrame):
    for key, value in metrics.items():
        if key not in df:
            df[key] = [value]
        else:
            df[key]._append(value)

@torch.no_grad()
def main():
    parser = HfArgumentParser((ScriptArguments))

    config: ScriptArguments = parser.parse_args_into_dataclasses()[0]

    set_seed(config.seed)

    sim_model = IntegerRegressionModel(config)

    dataset = Dataset.from_generator(get_dataset_generator, gen_kwargs={'num_examples': config.num_test_examples, 'model': sim_model, 'config': config})
    dataset.set_format(type='torch', columns=['prompt', 'answer'])

    model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=torch.bfloat16, device_map='cuda:0')
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    
    df = pd.DataFrame()
    
    loss = 0
    for i in range(max(1, len(dataset) - config.batch_size + 1)):
        batch = dataset[i:i+config.batch_size]
        outputs = prompt_model(model, tokenizer, batch['prompt'], config)
        outputs = torch.tensor([int(x) for x in outputs])
        targets = batch['answer']
        loss += evaluate(outputs, targets)
    loss /= len(dataset)
    metrics = {'loss': loss.item()}
    log_metrics(metrics, df)

    df.to_csv(f'{config.out_file}')

if __name__ == '__main__':
    main()
