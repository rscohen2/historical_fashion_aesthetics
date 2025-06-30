#!/bin/bash

DATA_PARTITION=$1

sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=debug
#SBATCH --partition=savio3_gpu
#SBATCH --account=fc_dbamman

#SBATCH --qos=savio_debug

#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

#SBATCH --gres=gpu:GTX2080TI:1


export DISTRIBUTED_RANK=1
export DISTRIBUTED_WORLD_SIZE=64
pixi run python -m src.fashion.pipeline.steps.extract_adjectives \
    --noun_mention_dir /global/scratch/users/naitian/fashion/data/pipeline/hathi_all/entity_mentions/$DATA_PARTITION/ \
    --output_dir /global/scratch/users/naitian/fashion/data/pipeline/hathi_all/adjectives/$DATA_PARTITION/ \
    --do_coref \
    --num_processes 64 \
    --concurrent_processes 1