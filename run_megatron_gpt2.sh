PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:24.01-py3
CHECKPOINT_PATH="/workspace/research/data/checkpoint" 
TENSORBOARD_LOGS_PATH="/workspace/research/data/tensorboard_logs"
VOCAB_FILE="/workspace/research/megatron/gpt2/vocab.json"
MERGE_FILE="/workspace/research/megatron/gpt2/merges.txt" #<Specify path to file>/gpt2-merges.txt
DATA_PATH="/workspace/research/megatron/wiki_gpt2_dataset_text_document" #<Specify path and file prefix>_text_document

docker run \
  --gpus=2 \
  --ipc=host \
  --workdir /workspace/research/megatron \
  -v /home/arvinj/Research:/workspace/research \
  -v /home/arvinj/Research/datasets_v2:/workspace/research/data \
  -v /home/arvinj/Research/Megatron-LM:/workspace/research/megatron \
  nvcr.io/nvidia/pytorch:24.01-py3 \
  bash /workspace/research/megatron/train_gpt2_megatron.sh $CHECKPOINT_PATH $TENSORBOARD_LOGS_PATH $VOCAB_FILE $MERGE_FILE $DATA_PATH