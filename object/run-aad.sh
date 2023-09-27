for ent_par in 1.0
do
#python AaD_src_pretrain.py --gpu_id 3  --net resnet50 --max_epoch 8 \
#--dset VISDA-C --s 0 --t 1 --run_name VISDA-C_source --sens_classes 1 --wandb \
#--da uda --output ckps/AaD/source/ --train_resampling natural --test_resampling natural

python AaD_src_pretrain.py --gpu_id 3  --net resnet50 --max_epoch 8 \
--dset cardiomegaly --s 0 --t 1 --run_name AaD_CheXpert_source --sens_classes 1 --wandb \
--da uda --output ckps/AaD/source/ --train_resampling natural --test_resampling natural

#python AaD_tar_adaptation.py --gpu_id 3 --net resnet50 --max_epoch 10 \
#--dset VISDA-C --s 0 --t 1 --run_name VISDA-C_T2V --sens_classes 1 --wandb \
#--da uda --output_src ckps/AaD/source/ --output ckps/AaD/target/ --train_resampling natural --test_resampling natural

python AaD_tar_adaptation.py --gpu_id 3 --net resnet50 --max_epoch 10 \
--dset cardiomegaly --s 0 --t 1 --run_name AaD_C2M --wandb --sens_classes 5 --wandb \
--da uda --output_src ckps/AaD/source/ --output ckps/AaD/target/ --train_resampling natural --test_resampling natural
done
