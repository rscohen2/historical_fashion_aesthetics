#!/bin/bash

DATA_PARTITION=$1

sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=debug
#SBATCH --partition=savio3_gpu
#SBATCH --account=fc_dbamman

#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

#SBATCH --gres=gpu:GTX2080TI:1

#SBATCH --output=/global/scratch/users/naitian/fashion/logs/s-%x.%A_%a.out
#SBATCH --error=/global/scratch/users/naitian/fashion/logs/s-%x.%A_%a.err


export DISTRIBUTED_RANK=1
export DISTRIBUTED_WORLD_SIZE=2
pixi run python -m src.fashion.pipeline.steps.extract_adjectives \
    --noun_mention_dir /global/scratch/users/naitian/fashion/data/pipeline/hathi_all/entity_mentions_dir/$DATA_PARTITION/ \
    --output_dir /global/scratch/users/naitian/fashion/data/pipeline/hathi_all/adjectives/$DATA_PARTITION/ \
    --do_coref \
    --num_processes 64 \
    --concurrent_processes 1
EOT