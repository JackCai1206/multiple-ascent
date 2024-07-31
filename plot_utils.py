from matplotlib import pyplot as plt
from random import choices, sample

from numpy import sqrt

from collections import defaultdict
from run import ScriptArguments, get_pz_list
from datasets import fingerprint
from transformers import HfArgumentParser, set_seed
import torch
from torch import Tensor
from typing import List, cast
from datasets.utils._dill import dumps
import os
from seaborn import color_palette

def pull_data(arg_list: List[ScriptArguments], get_key):
    configs = {}
    results = defaultdict(list)
    for config in arg_list:
        # set_seed(config.seed)
        fp_old = fingerprint.Hasher().hash(tuple(sorted(config.__dict__.items())))
        fp_new = repr(config)
        print(fp_old)
        for fp in [fp_old, fp_new]:
            folder = f'{config.out_dir}/cache/{fp}/'
            if os.path.exists(folder):
                break
        else:
            raise FileNotFoundError(f'No cache found for {config}')

        all_answers: Tensor = torch.load(f'{config.out_dir}/cache/{fp}/answers.pt')
        all_targets: Tensor = torch.load(f'{config.out_dir}/cache/{fp}/targets.pt')
        model_sd = torch.load(f'{config.out_dir}/cache/{fp}/model.pt')
        max_x_range = (config.x_range[1] - config.x_range[0]) * torch.ones((config.input_dim))
        mse = (all_answers - all_targets).to(torch.float64) / torch.dot(model_sd['linear.weight'].squeeze(), max_x_range) # (num_test_examples, len(pz_list)
        mse_pad = torch.cat([mse, torch.nan * torch.ones(config.num_test_examples - mse.shape[0], mse.shape[1])])
        key = get_key(config)
        results[key].append(mse_pad)
        configs[key] = config

    results = {k: torch.stack(results[k]) for k in results}
    return results, configs


def plot_2x2_avg(results: dict[tuple, Tensor], configs, x_list, w_list, logx=False, logy=True, vline=None, decompose_mse=False, pz_range=None):
    def remove_outliers(data, *others):
        return data[data <= 1], *[o[data <= 1] for o in others]

    fig = plt.figure(figsize=(6 * len(w_list), 4.5 * len(x_list)))
    axs = fig.subplots(len(x_list), len(w_list), squeeze=False)
    for col, avg_over_w in enumerate(w_list):
        for row, avg_over_x in enumerate(x_list):
            for ci, k in enumerate(results):
                # res_mask = (~results[k].isnan()).prod(dim=-1)
                res_mask = (~results[k].isnan()).sum(dim=-1) # we are just gonna allow for some nans
                # print(res_mask.shape)
                avail_w = torch.count_nonzero(res_mask, dim=1) >= avg_over_x
                avail_x = torch.count_nonzero(res_mask, dim=0) >= avg_over_w
                # avail = torch.outer(avail_x, avail_w)
                # print(avail_w, avail_x)
                i, j = torch.nonzero(avail_w).squeeze(-1).tolist(), torch.nonzero(avail_x).squeeze(-1).tolist()
                # print(i, j)
                if len(i) < avg_over_w or len(j) < avg_over_x:
                    print(f'Not enough data for {k}: {len(i)} out of {avg_over_w} weights, {len(j)} out of {avg_over_x} examples')
                    continue
                # print(avail_w, avail_x)
                # print(i, j)
                i, j = sample(i, k=avg_over_w), sample(j, k=avg_over_x)
                # print(i, j)
                dat = results[k][i, :, :][:, j, :]
                # dat = torch.log(dat)
                # print(results[k].shape, dat.shape)
                dat_bias = dat.flatten(0, 1).mean(dim=0).pow(2)
                dat_var = dat.flatten(0, 1).var(dim=0)
                dat_mse = dat.flatten(0, 1).pow(2).mean(dim=0)
                # if dat.shape[0] == 1 and dat.shape[1] == 1: # cannot compute variance
                #     dat_var = torch.zeros_like(dat_mean)
                # else:
                #     dat_var = dat.flatten(0, 1).std(dim=0)
                # print(dat.shape)
                labels = torch.tensor(get_pz_list(configs[k]))
                dat_mse, dat_bias, dat_var, labels = remove_outliers(dat_mse, dat_bias, dat_var, labels)
                if decompose_mse:
                    axs[row][col].plot(labels, dat_bias, linestyle='-', color=color_palette()[ci], label=k)
                    axs[row][col].plot(labels, dat_var, linestyle='--', color=color_palette()[ci])
                else:
                    axs[row][col].plot(labels, dat_mse, color=color_palette()[ci], label=k)
                # axs[row][col].fill_between(labels, dat_bias - dat_var, dat_bias + dat_var, alpha=0.2)
            if vline: axs[row][col].vlines(vline, 0, 1, color='red', linestyle='--')
            if pz_range is not None: axs[row][col].set_xlim(*pz_range)
            axs[row][col].set_xlabel('Prompt size')
            axs[row][col].set_ylabel('MSE')
            if logy:
                axs[row][col].semilogy()
                axs[row][col].set_ylim(1e-6, 1)
            if logx: axs[row][col].semilogx()
            axs[row][col].legend()
            axs[row][col].set_title(f'Prompt size vs MSE, average over {avg_over_w} weights, {avg_over_x} examples')

    plt.show()
