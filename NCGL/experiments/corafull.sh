# METHOD=ergnn
# METHOD=erreplace
# METHOD=dce
# METHOD=sl
# METHOD=our
# METHOD=ewc
# METHOD=mas
# METHOD=gem
# METHOD=twp
# METHOD=lwf
# METHOD=bare
METHOD=joint

CUDA_VISIBLE_DEVICES=0 python train.py \
--dataset CoraFull-CL \
--method $METHOD  \
--gpu 0 \
--ILmode classIL \
--inter-task-edges 'False' \
--minibatch 'False' \
--epochs 100 \
--ori_data_path data