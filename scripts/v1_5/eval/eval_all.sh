#!/bin/bash
#SBATCH --job-name=eval_viral
#SBATCH --output=/leonardo_scratch/large/userexternal/fgaragna/logs/%x-%j
#SBATCH --error=/leonardo_scratch/large/userexternal/fgaragna/logs/%x-%j
#SBATCH --open-mode=truncate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrB_MLLM-RAG
#SBATCH --array=0-1
#SBATCH --time=05:30:00

module load anaconda3/2022.05
module load profile/deeplrn
module load cuda/11.8
module unload gcc 
module load gcc/11.3.0

source activate viral
cd ~/viral
REPO_ROOT="$HOME/viral"
export PYTHONPATH="${REPO_ROOT}:$REPO_ROOT/lmms-eval:$PYTHONPATH"

export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=info
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HUB_CACHE="/leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/hf_models"
# export HF_DATASETS_CACHE="/leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/dataset/mllm_evaluation/cvprw"
export HF_DATASETS_CACHE="/leonardo_scratch/large/userexternal/fgaragna/dataset/mllm_evaluation/cvprw"
export HF_HOME="/leonardo_scratch/large/userexternal/fgaragna/dataset/mllm_evaluation/cvprw"
export HF_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

output_dir="/leonardo_scratch/large/userexternal/fgaragna/checkpoints/viral/${run_name}"

# task_list=(pope mme gqa scienceqa_img mmmu_val seedbench ai2d textvqa_val)
task_list=(pope mme)
echo ${task_list[$SLURM_ARRAY_TASK_ID]}

checkpoint_path="/leonardo_scratch/large/userexternal/fgaragna/checkpoints/viral/viral_llava_lora/"
base_model="/leonardo_scratch/large/userexternal/fgaragna/models/lmsys/vicuna-7b-v1.5"

srun -c $SLURM_CPUS_PER_TASK --mem $SLURM_MEM_PER_NODE \
python -u lmms-eval/lmms_eval/__main__.py \
--verbosity=DEBUG \
--task ${task_list[$SLURM_ARRAY_TASK_ID]} \
--model llava \
--model_args "pretrained=${checkpoint_path},attn_implementation=sdpa" \
--device cuda:0 \
--batch_size 1 \
--output /leonardo_scratch/large/userexternal/fgaragna/logs/lmms_eval \
--log_samples_suffix j \
--log_samples \
--timezone Europe/Paris