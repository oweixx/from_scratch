CUDA_VISIBLE_DEVICES=5 python train.py --name cifar10-100_500 --dataset cifar10 \
  --model_type ViT-L_16 --pretrained_dir ../models/ViT-L_16.npz \
  --fp16 --train_batch_size 128 --gradient_accumulation_steps 4
