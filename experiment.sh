set -e

# ----------------- Misc -----------------
# pz 2K -- 10*10 runs
# for seed in {0..9}; do
#     # for max_wx in 100 1000; do
#     python run.py --debug=False --api='togetherai' --seed=$seed --model_name='meta-llama/Meta-Llama-3-8B' --w_range 0 1000 --x_range 0 1000 --pz_end=600 --pz_start=2 --pz_dist=uniform --pz_count=100 --num_test_examples=10 --device='cuda:1'
#     python run.py --debug=False --api='togetherai' --seed=$seed --model_name='meta-llama/Meta-Llama-3-70B' --w_range 0 1000 --x_range 0 1000 --pz_end=600 --pz_start=2 --pz_dist=uniform --pz_count=100 --num_test_examples=10 --device='cuda:1'
#     python run.py --debug=False --api='togetherai' --seed=$seed --model_name='Qwen/Qwen1.5-32B' --w_range 0 1000 --x_range 0 1000 --pz_end=1000 --pz_start=2 --pz_dist=uniform --pz_count=100 --num_test_examples=10 --device='cuda:1'
# done


# ----------------- Meta-Llama 3.8B -----------------
# pz 0.6K -- 100 runs
# for seed in {0..100}; do
#     python run.py --debug=False --api='togetherai' --seed=$seed --model_name='meta-llama/Meta-Llama-3-8B' --w_range 0 1000 --x_range 0 1000 --pz_end=600 --pz_start=2 --pz_dist=log --pz_count=100 --num_test_examples=1 --device='cuda:1'
# done

# pz 2K -- 10*10 runs -- shuffle same context
# for seed in {0..2}; do
    # for max_wx in 100 1000; do
    # python run.py --debug=False --api='togetherai' --seed=$seed --model_name='meta-llama/Meta-Llama-3-8B' --w_range 0 1000 --x_range 0 1000 --pz_end=600 --pz_start=2 --pz_dist=uniform --pz_count=100 --num_test_examples=50 --dataset_type='shuffle' --device='cuda:1'
    # python run.py --debug=False --api='togetherai' --seed=$seed --model_name='meta-llama/Meta-Llama-3-70B' --w_range 0 1000 --x_range 0 1000 --pz_end=600 --pz_start=2 --pz_dist=uniform --pz_count=100 --num_test_examples=50 --dataset_type='shuffle' --device='cuda:1'
    # python run.py --use_cache=False --debug=False --api='baseline' --seed=$seed --model_name='KNN' --w_range 0 1000 --x_range 0 1000 --pz_end=600 --pz_start=2 --pz_dist=uniform --pz_count=100 --num_test_examples=50 --dataset_type='shuffle' --device='cuda:1'
# done

# ----------------- Haiku 3 -----------------
# pz 2K -- 6 * 10 runs
# for seed in {0..5}; do
#     python run.py --debug=False --api='anthropic' --seed=$seed --model_name='claude-3-haiku-20240307' --w_range 0 1000 --x_range 0 1000 --pz_end=2000 --pz_start=10 --pz_dist=log --pz_count=50 --num_test_examples=10 --device='cuda:1'
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
    python run.py --debug=False --api='anthropic' --seed=$seed --model_name='claude-3-5-sonnet-20240620' --w_range 0 1000 --x_range 0 1000 --pz_end=1000 --pz_start=2 --pz_dist=uniform --pz_count=50 --num_test_examples=30 --dataset_type='shuffle' --device='cuda:1'
done
