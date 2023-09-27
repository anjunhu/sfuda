for seed in 2023
do
#python NRC_train_src.py \
# --gpu_id 5 --net resnet50 --max_epoch 10 --seed $seed \
# --dset VISDA-C --s 0 --t 1 \
# --sens_classes 1 --wandb \
# --da uda --output ckps/NRC/source/ \
# --train_resampling natural --test_resampling natural

python NRC_train_tar.py \
 --gpu_id 5 --net resnet50 --max_epoch 10 --seed $seed \
 --dset VISDA-C --s 0 --t 1 \
 --sens_classes 1 --wandb \
 --da uda --output_src ckps/NRC/source/ --output ckps/NRC/target/ \
 --train_resampling natural --test_resampling natural
done
