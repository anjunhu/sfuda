#python DINE_dist.py --da uda --gpu_id 2 --dset VISDA-C \
# --train_resampling natural  --test_resampling natural --seed 2023 --sens_classes 1 \
# --net_src resnet50 --max_epoch 20  --output ckps/DINE/source/ --wandb

#python DINE_dist.py --da uda --gpu_id 2 --dset VISDA-C --distill \
# --train_resampling natural  --test_resampling natural --seed 2023 --sens_classes 1 \
# --topk 1 --net_src resnet50 --max_epoch 20  --output ckps/DINE/source/ --wandb

python DINE_ft.py --da uda --gpu_id 2 --dset VISDA-C \
 --train_resampling natural  --test_resampling natural --seed 2023 --sens_classes 1 \
 --net_src resnet50 --max_epoch 10 --net resnet50 --lr 1e-3 --output ./ckps/DINE/target
