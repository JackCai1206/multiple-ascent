set -e

python run.py --prompt_size=400 --num_test_examples=100 --prompt_size_step=10 --out_file=results/full.csv --debug=False --device='cuda:1'