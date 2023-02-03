# METHOD=bare
METHOD=ergnn
# METHOD=erreplace
# METHOD=dce
# METHOD=sl
# METHOD=our

CUDA_VISIBLE_DEVICES=0 python GCGL/train.py \
	--dataset SIDER-tIL \
	--method $METHOD \
	--backbone GCN \
	--gpu 0 \
	--clsIL 'True' \
	--num_epochs 100 \
	--result_path GCGL/results

