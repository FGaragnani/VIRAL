#!/bin/bash
#SBATCH --job-name=viral_llava_lora_dino
#SBATCH --output=/leonardo_scratch/large/userexternal/fgaragna/logs/%x-%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/fgaragna/logs/%x-%j.err
#SBATCH --open-mode=truncate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=480G
#SBATCH --cpus-per-task=32
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrB_MLLM-RAG
#SBATCH --time=24:00:00

module load anaconda3/2022.05
module load profile/deeplrn
module load cuda/11.8
module unload gcc 
module load gcc/11.3.0

source activate viral
cd ~/viral

REPO_ROOT="$HOME/viral"
export PYTHONPATH="${REPO_ROOT}:$PYTHONPATH"

export PYTHONUNBUFFERED=1
# export TORCH_HOME="/leonardo_scratch/large/userexternal/fgaragna/models/lmsys"
export TRANSFORMERS_VERBOSITY=info
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export WANDB_PROJECT=jeppetto
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HUB_CACHE="/leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/hf_models"
export HF_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

IFS=',' read -r -a nodelist <<<$SLURM_NODELIST
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=`comm -23 <(seq 5000 6000 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`

clip_model_name_or_path="openai/clip-vit-large-patch14"
pretrained_mm_mlp_adapter="./checkpoints/llava-v1.5-7b-pretrain/mm_projector.bin"
mm_projector_type="mlp2x_gelu"

learning_rate=2e-4
mm_projector_lr=2e-5
run_name="${SLURM_JOB_NAME}"
output_dir="/leonardo_scratch/large/userexternal/fgaragna/checkpoints/viral/${run_name}"

per_device_train_batch_size=16
gradient_accumulation_steps=2

language_model="/leonardo_scratch/large/userexternal/fgaragna/models/lmsys/vicuna-7b-v1.5"
train_data_path="/leonardo_scratch/large/userexternal/fgaragna/dataset/viral/llava_v1_5_mix665k.json"
train_image_folder="/leonardo_scratch/large/userexternal/fgaragna/dataset/viral"

((ws = $SLURM_NNODES * $SLURM_GPUS_PER_NODE))
export WORLD_SIZE=$ws

dataloader_num_workers=$(( $SLURM_CPUS_PER_TASK / $SLURM_GPUS_PER_NODE))

echo "Nodes: ${SLURM_NNODES}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "GPUs: ${SLURM_GPUS_PER_NODE}"
echo "MASTER ADDR: ${MASTER_ADDR}"
echo "MASTER PORT: ${MASTER_PORT}"
echo "WORLD SIZE: ${WORLD_SIZE}"
echo "DATALOADER WORKERS: ${dataloader_num_workers}"

srun --exclusive -c $SLURM_CPUS_PER_TASK --mem $SLURM_MEM_PER_NODE \
torchrun \
--nnodes=$SLURM_NNODES --nproc-per-node=$SLURM_GPUS_PER_NODE --rdzv-endpoint=$MASTER_ADDR --master-port=$MASTER_PORT --rdzv-id=$SLURM_JOB_NAME --rdzv-backend=c10d \
llava/train/train_mem.py \
--deepspeed ./scripts/zero2.json \
--lora_enable True \
--lora_r 128 \
--lora_alpha 256 \
--mm_projector_lr $mm_projector_lr \
--model_name_or_path $language_model \
--version v1 \
--data_path $train_data_path \
--image_folder $train_image_folder \
--vision_tower $clip_model_name_or_path \
--freeze_backbone True \
--pretrain_mm_mlp_adapter $pretrained_mm_mlp_adapter \
--mm_projector_type $mm_projector_type \
--mm_vision_select_layer -2 \
--mm_use_im_start_end False \
--mm_use_im_patch_token False \
--image_aspect_ratio pad \
--group_by_modality_length True \
--bf16 True \
--output_dir $output_dir \
--num_train_epochs 1 \
--per_device_train_batch_size $per_device_train_batch_size \
--per_device_eval_batch_size $per_device_train_batch_size \
--gradient_accumulation_steps $gradient_accumulation_steps \
--evaluation_strategy "no" \
--save_strategy steps \
--save_steps 1000 \
--learning_rate $learning_rate \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 5 \
--tf32 True \
--model_max_length 2048 \
--gradient_checkpointing True \
--dataloader_num_workers $dataloader_num_workers \
--seed 42 \
--run_name $run_name \
--lazy_preprocess True \
--report_to wandb \
--config_path ./config_2.json \
--use_glamm True \
--grand_image_dir "" \
--grand_annotation_dir ""