ngpu=$1
CUDA_VISIBLE_DEVICES=${ngpu} python test.py --dataset refcoco --savename savename --resume /home/ypf/workspace/code/BKINet/GLNet/saved_models/final_1021_model_best.pth.tar



