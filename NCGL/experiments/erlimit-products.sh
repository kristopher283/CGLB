CUDA_VISIBLE_DEVICES=2 python train.py \
--dataset Products-CL \
--method erlimit  \
--gpu 0 \
--ILmode classIL \
--inter-task-edges 'False' \
--minibatch 'True' \
--epochs 100 \
--ori_data_path data