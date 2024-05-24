set -e

# for prompt_size in 5 10 25 50 100 200 400 800; do
for prompt_size in $(seq 5 5 800); do
    export out_dir="results/num_examples_$prompt_size.csv"
    if [ -f out_dir ]; then
        echo "$outdir already exists, skipping"
        continue
    fi
    python run.py --prompt_size $prompt_size --out_file=$out_dir
done