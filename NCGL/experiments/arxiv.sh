# METHOD=ergnn
# METHOD=erreplace
METHOD=dce
# METHOD=sl
# METHOD=our

CUDA_VISIBLE_DEVICES=0 python train.py \
--dataset Arxiv-CL \
--method $METHOD  \
--gpu 0 \
--ILmode classIL \
--inter-task-edges 'False' \
--minibatch 'False' \
--epochs 100 \
--ori_data_path ./data