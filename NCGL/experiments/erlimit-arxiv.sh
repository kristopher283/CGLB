CUDA_VISIBLE_DEVICES=0 python train.py \
--dataset Arxiv-CL \
--method erlimit  \
--gpu 0 \
--ILmode classIL \
--inter-task-edges 'False' \
--minibatch 'False' \
--epochs 100 \
--ori_data_path data