for prompt_size in 5 10 25 50 100 200 400 800; do
    python run.py --prompt_size $prompt_size --out_file="results/num_examples_$num_examples"
done