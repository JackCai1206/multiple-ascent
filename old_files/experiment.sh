set -e

# ----------------- Misc -----------------
# pz 2K -- 10*10 runs
# for seed in {0..9}; do
#     # for max_wx in 100 1000; do
#     python run.py --debug=False --api='togetherai' --seed=$seed --model_name='meta-llama/Meta-Llama-3-8B' --w_range 0 1000 --x_range 0 1000 --pz_end=600 --pz_start=2 --pz_dist=uniform --pz_count=100 --num_test_examples=10 --device='cuda:1'
#     python run.py --debug=False --api='togetherai' --seed=$seed --model_name='meta-llama/Meta-Llama-3-70B' --w_range 0 1000 --x_range 0 1000 --pz_end=600 --pz_start=2 --pz_dist=uniform --pz_count=100 --num_test_examples=10 --device='cuda:1'
#     python run.py --debug=False --api='togetherai' --seed=$seed --model_name='Qwen/Qwen1.5-32B' --w_range 0 1000 --x_range 0 1000 --pz_end=1000 --pz_start=2 --pz_dist=uniform --pz_count=100 --num_test_examples=10 --device='cuda:1'
# done

# ----------------- Baselines -----------------
# pz 0.6K -- 10 * 100 runs
# for seed in {0..9}; do
#     python run.py --use_cache=False --debug=False --api='baseline' --seed=$seed --model_name='KNN-1' --w_range 0 1000 --x_range 0 1000 --pz_end=600 --pz_start=1 --pz_dist=uniform --pz_count=100 --num_test_examples=100 --device='cuda:1'
#     python run.py --use_cache=False --debug=False --api='baseline' --seed=$seed --model_name='KNN-2' --w_range 0 1000 --x_range 0 1000 --pz_end=600 --pz_start=2 --pz_dist=uniform --pz_count=100 --num_test_examples=100 --device='cuda:1'
#     python run.py --use_cache=False --debug=False --api='baseline' --seed=$seed --model_name='KNN-5' --w_range 0 1000 --x_range 0 1000 --pz_end=600 --pz_start=5 --pz_dist=uniform --pz_count=100 --num_test_examples=100 --device='cuda:1'
#     python run.py --use_cache=False --debug=False --api='baseline' --seed=$seed --model_name='KNN-10' --w_range 0 1000 --x_range 0 1000 --pz_end=600 --pz_start=10 --pz_dist=uniform --pz_count=100 --num_test_examples=100 --device='cuda:1'
#     python run.py --use_cache=False --debug=False --api='baseline' --seed=$seed --model_name='KNN-20' --w_range 0 1000 --x_range 0 1000 --pz_end=600 --pz_start=20 --pz_dist=uniform --pz_count=100 --num_test_examples=100 --device='cuda:1'
# done

# pz 2K -- 10 * 100 runs
# for seed in {0..9}; do
#     python run.py --use_cache=False --debug=False --api='baseline' --seed=$seed --model_name='KNN-1' --w_range 0 1000 --x_range 0 1000 --pz_end=2000 --pz_start=1 --pz_dist=uniform --pz_count=100 --num_test_examples=100 --device='cuda:1'
#     python run.py --use_cache=False --debug=False --api='baseline' --seed=$seed --model_name='KNN-2' --w_range 0 1000 --x_range 0 1000 --pz_end=2000 --pz_start=2 --pz_dist=uniform --pz_count=100 --num_test_examples=100 --device='cuda:1'
#     python run.py --use_cache=False --debug=False --api='baseline' --seed=$seed --model_name='KNN-5' --w_range 0 1000 --x_range 0 1000 --pz_end=2000 --pz_start=5 --pz_dist=uniform --pz_count=100 --num_test_examples=100 --device='cuda:1'
#     python run.py --use_cache=False --debug=False --api='baseline' --seed=$seed --model_name='KNN-10' --w_range 0 1000 --x_range 0 1000 --pz_end=2000 --pz_start=10 --pz_dist=uniform --pz_count=100 --num_test_examples=100 --device='cuda:1'
#     python run.py --use_cache=False --debug=False --api='baseline' --seed=$seed --model_name='KNN-20' --w_range 0 1000 --x_range 0 1000 --pz_end=2000 --pz_start=20 --pz_dist=uniform --pz_count=100 --num_test_examples=100 --device='cuda:1'
# done

# pz 4K -- 10 * 100 runs - dim 4
# for seed in {0..9}; do
#     python run.py --use_cache=False --debug=False --api='baseline' --seed=$seed --model_name='KNN-1' --w_range 0 1000 --x_range 0 1000 --input_dim=4 --pz_end=4000 --pz_start=1 --pz_dist=uniform --pz_count=100 --num_test_examples=100 --device='cuda:1'
#     python run.py --use_cache=False --debug=False --api='baseline' --seed=$seed --model_name='KNN-2' --w_range 0 1000 --x_range 0 1000 --input_dim=4 --pz_end=4000 --pz_start=2 --pz_dist=uniform --pz_count=100 --num_test_examples=100 --device='cuda:1'
#     python run.py --use_cache=False --debug=False --api='baseline' --seed=$seed --model_name='KNN-5' --w_range 0 1000 --x_range 0 1000 --input_dim=4 --pz_end=4000 --pz_start=5 --pz_dist=uniform --pz_count=100 --num_test_examples=100 --device='cuda:1'
#     python run.py --use_cache=False --debug=False --api='baseline' --seed=$seed --model_name='KNN-10' --w_range 0 1000 --x_range 0 1000 --input_dim=4 --pz_end=4000 --pz_start=10 --pz_dist=uniform --pz_count=100 --num_test_examples=100 --device='cuda:1'
#     python run.py --use_cache=False --debug=False --api='baseline' --seed=$seed --model_name='KNN-20' --w_range 0 1000 --x_range 0 1000 --input_dim=4 --pz_end=4000 --pz_start=20 --pz_dist=uniform --pz_count=100 --num_test_examples=100 --device='cuda:1'
# done

# pz 4K -- 10 * 100 runs - dim 6
# for seed in {0..9}; do
#     python run.py --use_cache=False --debug=False --api='baseline' --seed=$seed --model_name='KNN-1' --w_range 0 1000 --x_range 0 1000 --input_dim=6 --pz_end=4000 --pz_start=1 --pz_dist=uniform --pz_count=100 --num_test_examples=100 --device='cuda:1'
#     python run.py --use_cache=False --debug=False --api='baseline' --seed=$seed --model_name='KNN-2' --w_range 0 1000 --x_range 0 1000 --input_dim=6 --pz_end=4000 --pz_start=2 --pz_dist=uniform --pz_count=100 --num_test_examples=100 --device='cuda:1'
#     python run.py --use_cache=False --debug=False --api='baseline' --seed=$seed --model_name='KNN-5' --w_range 0 1000 --x_range 0 1000 --input_dim=6 --pz_end=4000 --pz_start=5 --pz_dist=uniform --pz_count=100 --num_test_examples=100 --device='cuda:1'
#     python run.py --use_cache=False --debug=False --api='baseline' --seed=$seed --model_name='KNN-10' --w_range 0 1000 --x_range 0 1000 --input_dim=6 --pz_end=4000 --pz_start=10 --pz_dist=uniform --pz_count=100 --num_test_examples=100 --device='cuda:1'
#     python run.py --use_cache=False --debug=False --api='baseline' --seed=$seed --model_name='KNN-20' --w_range 0 1000 --x_range 0 1000 --input_dim=6 --pz_end=4000 --pz_start=20 --pz_dist=uniform --pz_count=100 --num_test_examples=100 --device='cuda:1'
# done

# ----------------- Meta-Llama 3 8B -----------------
# pz 0.6K -- 100 runs
# for seed in {0..100}; do
#     python run.py --debug=False --api='togetherai' --seed=$seed --model_name='meta-llama/Meta-Llama-3-8B' --w_range 0 1000 --x_range 0 1000 --pz_end=600 --pz_start=2 --pz_dist=log --pz_count=100 --num_test_examples=1 --device='cuda:1'
# done

# pz 0.6K -- 10*10 runs -- shuffle same context
# for seed in {0..2}; do
    # for max_wx in 100 1000; do
    # python run.py --debug=False --api='togetherai' --seed=$seed --model_name='meta-llama/Meta-Llama-3-8B' --w_range 0 1000 --x_range 0 1000 --pz_end=600 --pz_start=2 --pz_dist=uniform --pz_count=100 --num_test_examples=50 --dataset_type='shuffle' --device='cuda:1'
    # python run.py --debug=False --api='togetherai' --seed=$seed --model_name='meta-llama/Meta-Llama-3-70B' --w_range 0 1000 --x_range 0 1000 --pz_end=600 --pz_start=2 --pz_dist=uniform --pz_count=100 --num_test_examples=50 --dataset_type='shuffle' --device='cuda:1'
    # python run.py --use_cache=False --debug=False --api='baseline' --seed=$seed --model_name='KNN' --w_range 0 1000 --x_range 0 1000 --pz_end=600 --pz_start=2 --pz_dist=uniform --pz_count=100 --num_test_examples=50 --dataset_type='shuffle' --device='cuda:1'
# done

# pz 0.6K -- 10 * 20 runs
# for seed in {0..9}; do
#     python run.py --use_cache=False --debug=False --api='togetherai' --seed=$seed --model_name='meta-llama/Meta-Llama-3-70B' --w_range 0 1000 --x_range 0 1000 --pz_end=600 --pz_start=2 --pz_dist=uniform --pz_count=20 --num_test_examples=20 --device='cuda:1'
# done

# pz 0.6K -- 1 * 100 runs
# for seed in 0; do
#     python run.py --debug=False --api='togetherai' --seed=$seed --model_name='meta-llama/Meta-Llama-3-8B' --w_range 0 1000 --x_range 0 1000 --pz_end=600 --pz_start=2 --pz_dist=uniform --pz_count=40 --num_test_examples=100 --device='cuda:1'
#     python run.py --debug=False --api='togetherai' --seed=$seed --model_name='meta-llama/Meta-Llama-3-70B' --w_range 0 1000 --x_range 0 1000 --pz_end=600 --pz_start=2 --pz_dist=uniform --pz_count=40 --num_test_examples=100 --device='cuda:1'
# done

# pz 0.35K -- 3 * 100 runs - dim 4
# for seed in {0..2}; do
#     python run.py --debug=False --api='togetherai' --seed=$seed --model_name='meta-llama/Meta-Llama-3-8B' --w_range 0 1000 --x_range 0 1000 --input_dim=4 --pz_end=350 --pz_start=2 --pz_dist=uniform --pz_count=40 --num_test_examples=100 --device='cuda:1'
#     python run.py --debug=False --api='togetherai' --seed=$seed --model_name='meta-llama/Meta-Llama-3-70B' --w_range 0 1000 --x_range 0 1000 --input_dim=4 --pz_end=350 --pz_start=2 --pz_dist=uniform --pz_count=40 --num_test_examples=100 --device='cuda:1'
# done

# pz 0.3K -- 3 * 100 runs - dim 6
# for seed in {0..2}; do
#     python run.py --debug=False --api='togetherai' --seed=$seed --model_name='meta-llama/Meta-Llama-3-8B' --w_range 0 1000 --x_range 0 1000 --input_dim=6 --pz_end=300 --pz_start=2 --pz_dist=uniform --pz_count=40 --num_test_examples=100 --device='cuda:1'
#     python run.py --debug=False --api='togetherai' --seed=$seed --model_name='meta-llama/Meta-Llama-3-70B' --w_range 0 1000 --x_range 0 1000 --input_dim=6 --pz_end=300 --pz_start=2 --pz_dist=uniform --pz_count=40 --num_test_examples=100 --device='cuda:1'
# done

# ----------------- Haiku 3 -----------------
# pz 2K -- 6 * 10 runs
# for seed in {0..5}; do
#     python run.py --debug=False --api='anthropic' --seed=$seed --model_name='claude-3-haiku-20240307' --w_range 0 1000 --x_range 0 1000 --pz_end=2000 --pz_start=10 --pz_dist=log --pz_count=50 --num_test_examples=10 --device='cuda:1'
# done

# pz 2K -- 1 * 100 runs
# for seed in 0; do
#     python run.py --debug=False --api='anthropic' --seed=$seed --model_name='claude-3-haiku-20240307' --w_range 0 1000 --x_range 0 1000 --pz_end=2000 --pz_start=10 --pz_dist=log --pz_count=50 --num_test_examples=100 --device='cuda:1'
# done

# pz 16K -- 5 * 5 runs -- xw range 100
# for seed in {0..5}; do
#     python run.py --debug=False --api='anthropic' --seed=$seed --model_name='claude-3-haiku-20240307' --w_range 0 100 --x_range 0 100 --pz_end=16000 --pz_start=10 --pz_dist=log --pz_count=50 --num_test_examples=5 --device='cuda:1'
# done

# ----------------- Sonnet 3.5 -----------------
# pz 2K -- 6 * 5 runs
# for seed in {0..5}; do
#     python run.py --debug=False --api='anthropic' --seed=$seed --model_name='claude-3-5-sonnet-20240620' --w_range 0 1000 --x_range 0 1000 --pz_end=2000 --pz_start=2 --pz_dist=log --pz_count=50 --num_test_examples=5 --device='cuda:1'
# done

# pz 16K -- 6 * 5 runs
# for seed in {0..5}; do
#     python run.py --debug=False --api='anthropic' --seed=$seed --model_name='claude-3-5-sonnet-20240620' --w_range 0 1000 --x_range 0 1000 --pz_end=16000 --pz_start=2 --pz_dist=log --pz_count=50 --num_test_examples=5 --device='cuda:1'
# done

# pz 2K -- 1 * 30 runs -- shuffle same context
for seed in 0; do
    python -m pdb run.py --debug=False --api='anthropic' --seed=$seed --model_name='claude-3-5-sonnet-20240620' --w_range 0 1000 --x_range 0 1000 --pz_end=1000 --pz_start=2 --pz_dist=uniform --pz_count=50 --num_test_examples=30 --dataset_type='shuffle' --device='cuda:1'
done

# pz 1.5K -- 1 * 100 runs - dim 4
# for seed in 0; do
#     python run.py --debug=False --api='anthropic' --seed=$seed --model_name='claude-3-5-sonnet-20240620' --w_range 0 1000 --x_range 0 1000 --input_dim=4 --pz_end=1500 --pz_start=2 --pz_dist=uniform --pz_count=40 --num_test_examples=100 --device='cuda:1'
# done

# pz 1.3K -- 1 * 100 runs - dim 6
# for seed in 0; do
#     python run.py --debug=False --api='anthropic' --seed=$seed --model_name='claude-3-5-sonnet-20240620' --w_range 0 1000 --x_range 0 1000 --input_dim=6 --pz_end=1300 --pz_start=2 --pz_dist=uniform --pz_count=40 --num_test_examples=100 --device='cuda:1'
# done
