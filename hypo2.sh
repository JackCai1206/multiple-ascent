set -e 

# Compare with and without instruction tuning
for seed in {0..2}; do
    python run.py --debug=False --api='togetherai' --seed=$seed --model_name='meta-llama/Meta-Llama-3-70B' --w_range 0 10 --x_range 0 1000 --input_dim=3 --pz_end=540 --pz_start=2 --pz_dist=uniform --pz_count=50 --num_test_examples=30 --dataset_type='default' --device='cuda:1'
    python run.py --debug=False --api='togetherai' --seed=$seed --model_name='meta-llama/Meta-Llama-3-70B-Instruct-Turbo' --w_range 0 10 --x_range 0 1000 --input_dim=3 --pz_end=540 --pz_start=2 --pz_dist=uniform --pz_count=50 --num_test_examples=30 --dataset_type='default' --device='cuda:1'
    python run.py --use_cache=False --debug=False --api='baseline' --seed=$seed --model_name='KNN' --w_range 0 10 --x_range 0 1000 --input_dim=3 --pz_end=1000 --pz_start=2 --pz_dist=uniform --pz_count=100 --num_test_examples=100 --dataset_type='default' --device='cuda:1'
done
