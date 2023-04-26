export CUDA_VISIBLE_DEVICES=0
python train.py \
--respath checkpoints/BiSeSTDC_seg/ \
--backbone BiSeSTDCNet \
--mode train \
--n_workers_train 0 \
--n_workers_val 1 \
--max_iter 160000 \
--n_img_per_gpu 6 \
--use_boundary_8 true \
--pretrain_path checkpoints/BiSeSTDCNePre.tar

