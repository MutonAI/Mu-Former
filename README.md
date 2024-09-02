# Introduction 
The repository primarily hosts the code for μFormer, or Mu-Former, uFormer, Muformer for readability, a potent tool tailored for predicting the effects of protein mutations. It is configured to facilitate the replication of the models presented in the paper titled *Accelerating protein engineering with fitness landscape modeling and reinforcement learning* which can be accessed at [this link](https://www.biorxiv.org/content/10.1101/2023.11.16.565910v2). Please note that the official release of this repository is expected soon. It will receive ongoing updates and maintenance. After its release, this current repository will be retired.


# Environment

To ensure optimal functioning of the uFormer application, a specific Conda environment should be set up. We've tested this setup using CUDA Version 12.2. Follow the steps below to set up the Conda environment:
```
conda create -n mutation python==3.8  
conda activate mutation  
pip install -r requirements.txt  
```

Additionally, you need to install PyTorch. The version to be installed is dependent on your GPU driver version. For instance:
```
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
```
Or for a cpu-only version:
```
pip install torch==1.12.0+cpu torchvision==0.13.0+cpu torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cpu
```

# Getting Started

The pre-trained encoder and sample datasets for fine-tuning are publicly accessible on [figshare](https://doi.org/10.6084/m9.figshare.26892355). Begin by downloading the checkpoint and sample datasets to your local storage. Subsequently, you can follow the provided command lines to fine-tune the model for your data, using the pre-prepared encoder.

If you're using a single GPU card, you can run the application using the following command:

```
python main.py --decoder-name mono --encoder-lr 1e-5 --decoder-lr 1e-4 \
  --epochs 300 --warmup-epochs 10 --batch-size 8 \
  --pretrained-model <path/to/uformer_l_encoder.pt> \
  --fasta <path/to/fasta>  \
  --train <path/to/train/tsv>  \
  --valid <path/to/valid/tsv>  \
  --test <path/to/test/tsv>  \
  --output-dir <path/to/existing/dir>
```

In case you're running the program on a node with multiple GPU cards (4, for example), the command can be adjusted as follows:
```
python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --node_rank=0 --master_port=6005 \
  main.py --decoder-name mono --encoder-lr 1e-5 --decoder-lr 1e-4 --batch-size 2 \
  --epochs 300 --warmup-epochs 10 \
  --pretrained-model <path/to/uformer_l_encoder.pt> \
  --fasta <path/to/fasta>  \
  --train <path/to/train/tsv>  \
  --valid <path/to/valid/tsv>  \
  --test <path/to/test/tsv>  \
  --output-dir <path/to/existing/dir>
```
