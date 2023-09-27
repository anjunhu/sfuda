python TENT_target.py --da uda --gpu_id 4 --dset cardiomegaly \
 --train_resampling natural  --test_resampling natural --seed 2023 --sens_classes 5 \
 --output_src ckps/SHOT/source/ --output ckps/TENT/target/  --s 0 --t 1 --wandb

#python TENT_target.py --da uda --gpu_id 3 --dset VISDA-C \
# --train_resampling natural  --test_resampling natural --seed 2023 --sens_classes 1 \
# --output_src ckps/SHOT/source/ --output ckps/TENT/target/  --s 0 --t 1 --wandb
