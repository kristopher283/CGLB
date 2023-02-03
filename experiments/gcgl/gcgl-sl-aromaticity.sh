CUDA_VISIBLE_DEVICES=1 python GCGL/train.py \
	--dataset Aromaticity-CL \
	--method sl \
	--backbone GCN \
	--gpu 0 \
	--clsIL True \
	--num_epochs 50 \
	--result_path GCGL/results
