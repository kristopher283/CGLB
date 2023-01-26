METHOD=bare
# METHOD=independent
# METHOD=joint
# METHOD=ergnn
# METHOD=erreplace
# METHOD=dce
# METHOD=sl
# METHOD=our

CUDA_VISIBLE_DEVICES=1 python GCGL/train.py \
	--dataset ENZYMES-CL \
	--method $METHOD \
	--backbone GCN \
	--gpu 0 \
	--clsIL 'True' \
	--num_epochs 100 \
	--result_path GCGL/results \
	--repeats 5

