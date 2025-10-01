pip install .
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
CUDA_VISIBLE_DEVICES=0 python scripts/train_cm/train_ecm.py exp.name=cifar10_ecm exp.use_wandb=False dataset.batch_size=64 unet.augment_dim=0 unet.ckpt_path=./pretrained/edm-cifar10-32x32-uncond-vp.pkl