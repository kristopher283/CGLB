# METHOD=ergnn
# METHOD=erreplace
# METHOD=dce
# METHOD=sl
# METHOD=our
METHOD=joint

CUDA_VISIBLE_DEVICES=3 python train.py \
--dataset Products-CL \
--method $METHOD  \
--gpu 0 \
--ILmode classIL \
--inter-task-edges 'False' \
--minibatch 'True' \
--epochs 100 \
--ori_data_path data