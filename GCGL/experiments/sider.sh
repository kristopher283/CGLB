METHOD=ergnn

CUDA_VISIBLE_DEVICES=0 python GCGL/train.py \
	--dataset SIDER-tIL \
	--method $METHOD \
	--backbone GCN \
	--gpu 0 \
	--clsIL 'False' \
	--num_epochs 100

