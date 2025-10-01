pip install .
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
CUDA_VISIBLE_DEVICES=0 python scripts/train_clf/train_clf.py exp.name=cifar10_clf exp.use_wandb=False aug=geo_clr_deg classifier.n_classes=15 dataset.batch_size=512