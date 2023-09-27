#python SHOT_source.py --output ckps/SHOT/source/ --da uda --gpu_id 4 --dset VISDA-C \
# --train_resampling natural  --test_resampling natural \
# --max_epoch 10 --s 1 --t 1 --wandb --seed 2023 --sens_classes 1

python SHOT_target.py --cls_par 0.3 --da uda --gpu_id 0 --dset cardiomegaly \
 --train_resampling natural  --test_resampling natural --seed 2023 --sens_classes 5 --gent \
 --output_src ckps/SHOT/source/ --output ckps/SHOT/target/  --s 1 --t 0 --run_name Cardiomegaly_M2C  --wandb
