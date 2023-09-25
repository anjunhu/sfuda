#python SHOT_source.py --output ckps/SHOT/source/ --da uda --gpu_id 4 --dset VISDA-C \
# --train_resampling natural  --test_resampling natural \
# --max_epoch 10 --s 0 --t 0 --wandb --seed 2023 --sens_classes 1

python SHOT_target.py --cls_par 0.3 --da uda --gpu_id 4 --dset VISDA-C \
 --train_resampling natural  --test_resampling natural --seed 2023 --sens_classes 1 \
 --output_src ckps/SHOT/source/ --output ckps/SHOT/target/  --s 0 --t 1 --run_name VISDA_T2V  --wandb
