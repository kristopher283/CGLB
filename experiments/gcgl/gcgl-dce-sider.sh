CUDA_VISIBLE_DEVICES=0 python3 GCGL/train.py \
	--dataset SIDER-tIL \
	--method dce \
	--backbone GCN \
	--gpu 0 \
	--clsIL True \
	--num_epochs 100
