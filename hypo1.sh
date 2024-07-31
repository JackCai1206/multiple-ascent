set -e

# Baseline
for seed in {0..2}; do
    python run.py --debug=False --api='anthropic' --seed=$seed --model_name='claude-3-5-sonnet-20240620' --w_range 0 10 --x_range 0 1000 --input_dim=3 --pz_end=1000 --pz_start=2 --pz_dist=uniform --pz_count=50 --num_test_examples=30 --dataset_type='default' --device='cuda:1'
    python run.py --use_cache=False --debug=False --api='baseline' --seed=$seed --model_name='KNN' --w_range 0 10 --x_range 0 1000 --input_dim=3 --pz_end=1000 --pz_start=2 --pz_dist=uniform --pz_count=100 --num_test_examples=100 --dataset_type='default' --device='cuda:1'
done

# Is the context-induced function sensitive to example order? 
for seed in {0..2}; do
    python run.py --debug=False --api='anthropic' --seed=$seed --model_name='claude-3-5-sonnet-20240620' --w_range 0 10 --x_range 0 1000 --input_dim=3 --pz_end=1000 --pz_start=2 --pz_dist=uniform --pz_count=50 --num_test_examples=30 --dataset_type='shuffle' --device='cuda:1'
    python run.py --use_cache=False --debug=False --api='baseline' --seed=$seed --model_name='KNN' --w_range 0 10 --x_range 0 1000 --input_dim=3 --pz_end=1000 --pz_start=2 --pz_dist=uniform --pz_count=100 --num_test_examples=100 --dataset_type='shuffle' --device='cuda:1'
done
