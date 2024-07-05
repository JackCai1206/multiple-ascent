set -e

# python run.py --prompt_size=600 --num_test_examples=1000 --prompt_size_step=10 --out_file=results/full.csv --debug=False --device='cuda:1'
for seed in {0..9}; do
    # for max_wx in 100 1000; do
    python run.py --debug=False --api='togetherai' --seed=$seed --model_name='meta-llama/Meta-Llama-3-8B' --max_w=1000 --max_x=1000 --pz_end=600 --pz_start=2 --pz_dist=uniform --pz_count=100 --num_test_examples=100 --device='cuda:1'
    python run.py --debug=False --api='togetherai' --seed=$seed --model_name='meta-llama/Meta-Llama-3-70B' --max_w=1000 --max_x=1000 --pz_end=600 --pz_start=2 --pz_dist=uniform --pz_count=100 --num_test_examples=100 --device='cuda:1'
    python run.py --debug=False --api='togetherai' --seed=$seed --model_name='Qwen/Qwen1.5-32B' --max_w=1000 --max_x=1000 --pz_end=1000 --pz_start=2 --pz_dist=uniform --pz_count=100 --num_test_examples=100 --device='cuda:1'
done

# python run.py --debug=False --api='anthropic' --model_name='claude-3-haiku-20240307' --pz_end=2000 --pz_start=10 --pz_dist=log --pz_count=100 --num_test_examples=5 --out_file=results/full.csv --device='cuda:1'
# python run.py --debug=False --api='anthropic' --model_name='claude-3-5-sonnet-20240620' --pz_end=2000 --pz_start=10 --pz_dist=log --pz_count=100 --num_test_examples=5 --out_file=results/full.csv --device='cuda:1'
