METHOD=ergnn

CUDA_VISIBLE_DEVICES=3 python GCGL/train.py \
	--dataset Aromaticity-CL \
	--method $METHOD \
	--backbone GCN \
	--gpu 0 \
	--clsIL 'True' \
	--num_epochs 100 \
	--result_path GCGL/results

