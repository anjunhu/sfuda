export CUDA_VISIBLE_DEVICES=0,1,2,3,5

# train source model
PORT=10000
SRC_MODEL_DIR=/scratch/local/ssd/anjun/sfuda/SHOT/object/ckps/AdaContrast/VISDA-C/source/

for SEED in 2023
do

echo $SRC_MODEL_DIR
echo $SEED

#    MEMO="source"
#    python AdaContrast_main.py train_source=true learn=source \
#    seed=${SEED} port=${PORT} memo=${MEMO} project="VISDA-C" \
#    data.data_root="${PWD}/data" data.workers=8 \
#    data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
#    learn.epochs=10 \
#    model_src.arch="resnet50" \
#    optim.lr=2e-4

#    MEMO="target"
#    python AdaContrast_main.py \
#    seed=${SEED} port=${PORT} memo=${MEMO} project="VISDA-C" \
#    data.data_root="${PWD}/data" data.workers=8  data.sens_classes=1 \
#    data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
#    model_src.arch="resnet50" \
#    model_tta.src_log_dir=${SRC_MODEL_DIR} \
#    optim.lr=2e-4

done

SRC_MODEL_DIR=/scratch/local/ssd/anjun/sfuda/SHOT/object/ckps/AdaContrast/cardiomegaly/source/

for SEED in 2023
do

echo $SRC_MODEL_DIR
echo $SEED

#    MEMO="source"
#    python AdaContrast_main.py train_source=true learn=source \
#    seed=${SEED} port=${PORT} memo=${MEMO} project="AdaContrast" \
#    data.data_root="${PWD}/data" data.workers=8   data.sens_classes=5 \
#    data.dataset="cardiomegaly" data.source_domains="[chexpert]" data.target_domains="[mimic]" \
#    learn.epochs=10 \
#    model_src.arch="resnet50" \
#    optim.lr=2e-4

#    MEMO="target"
#    python AdaContrast_main.py \ 
#    seed=${SEED} port=${PORT} memo=${MEMO} project="AdaContrast" \
#    data.data_root="${PWD}/data" data.workers=8  data.sens_classes=1 \ 
#    data.dataset="cardiomegaly" data.source_domains="[chexpert]" data.target_domains="[mimic]" \
#    model_src.arch="resnet50" \ 
#    model_tta.src_log_dir=${SRC_MODEL_DIR} \ 
#    optim.lr=2e-4 

done

