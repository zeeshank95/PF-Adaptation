
export ROOT=$(pwd)

MAX_TOKENS=3050
LR=3e-4
UPDATE_FREQ=2
CHECKPOINTS=checkpoints


#Train the Parent MNMT model. --dataset-name = multi, --dataset-idx = 1, use flag --init_multi

function train_MNMT {
	MAX_EPOCH=35
	python train.py config.yaml \
	    --task shared-multilingual-translation \
	    --dataset-name multi \
	    --dataset-idx 1 \
	    --bilingual \
	    --init-multi \
	    --max-epoch $MAX_EPOCH \
	    --num-workers 0 \
	    --arch transformer \
	    --max-tokens $MAX_TOKENS \
	    --lr $LR \
	    --min-lr 1e-9 \
	    --optimizer adam --adam-betas '(0.9, 0.98)' \
	    --save-dir $CHECKPOINTS \
	    --log-format simple \
	    --log-interval 10 \
	    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
	    --dropout 0.3 --weight-decay 0.0001 \
	    --attention-dropout 0.1 \
	    --activation-dropout 0.1 \
	    --ddp-backend no_c10d \
	    --update-freq $UPDATE_FREQ \
	    --share-all-embeddings \
	    --lr-scheduler inverse_sqrt \
	    --warmup-init-lr 1e-07 \
	    --warmup-updates 4500 \
	    --tensorboard-logdir logs \
	    --save-interval-updates 1000
}

#Prune the Parent MNMT and retrain it for 5 more epochs to compensate for heavy pruning. 
# --dataset-name = multi, --dataset-idx = 1 use flag --prune

function prune_MNMT {
	MAX_EPOCH=40
	python train.py config.yaml \
	--task shared-multilingual-translation \
	--dataset-name multi \
	--dataset-idx 1 \
	--bilingual \
	--prune \
	--prune-perc 0.50 \
	--max-epoch $MAX_EPOCH \
	--num-workers 0 \
	--arch transformer \
	--max-tokens $MAX_TOKENS \
	--lr $LR \
	--min-lr 1e-9 \
	--optimizer adam --adam-betas '(0.9, 0.98)' \
	--save-dir $CHECKPOINTS \
	--log-format simple \
	--log-interval 10 \
	--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
	--dropout 0.3 --weight-decay 0.0001 \
	--attention-dropout 0.1 \
	--activation-dropout 0.1 \
	--ddp-backend no_c10d \
	--update-freq $UPDATE_FREQ \
	--share-all-embeddings \
	--lr-scheduler inverse_sqrt \
	--warmup-init-lr 1e-07 \
	--warmup-updates 1000 \
	--restore-file checkpoints/multi_1_baseline/checkpoint_last.pt
}

# Train the free parameters for language 1 i.e. Arabic in this case, for 20 epochs
# --dataset-name = ar, --dataset-idx = 2 use flag --finetune

function fine_ar {
    MAX_EPOCH=60 # Also change this line
    python train.py config.yaml \
    --task shared-multilingual-translation \
    --dataset-name ar \
    --dataset-idx 2 \
    --bilingual \
    --finetune \
    --max-epoch $MAX_EPOCH \
    --num-workers 0 \
    --arch transformer \
    --max-tokens $MAX_TOKENS \
    --lr $LR \
    --min-lr 1e-9 \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --save-dir $CHECKPOINTS \
    --log-format simple \
    --log-interval 10 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --dropout 0.3 --weight-decay 0.0001 \
    --attention-dropout 0.1 \
    --activation-dropout 0.1 \
    --ddp-backend no_c10d \
    --update-freq $UPDATE_FREQ \
    --share-all-embeddings \
    --lr-scheduler inverse_sqrt \
    --warmup-init-lr 1e-07 \
    --warmup-updates 1000 \
    --reset-optimizer \
    --reset-lr-scheduler \
    --restore-file $CHECKPOINTS/multi_1_final/checkpoint_last.pt
}


#Prune the parameters of the current language pair only and retrain it for 10 epochs to compensate for heavy pruning
# --dataset-name = ar, --dataset-idx = 2 use flag --prune
function prune_ar {
	MAX_EPOCH=70
	python train.py config.yaml \
	--task shared-multilingual-translation \
	--dataset-name ar \
	--dataset-idx 2 \
	--bilingual \
	--prune \
	--prune-perc 0.75 \
	--max-epoch $MAX_EPOCH \
	--num-workers 0 \
	--arch transformer \
	--max-tokens $MAX_TOKENS \
	--lr $LR \
	--min-lr 1e-9 \
	--optimizer adam --adam-betas '(0.9, 0.98)' \
	--save-dir $CHECKPOINTS \
	--log-format simple \
	--log-interval 10 \
	--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
	--dropout 0.3 --weight-decay 0.0001 \
	--attention-dropout 0.1 \
	--activation-dropout 0.1 \
	--ddp-backend no_c10d \
	--update-freq $UPDATE_FREQ \
	--share-all-embeddings \
	--lr-scheduler inverse_sqrt \
	--warmup-init-lr 1e-07 \
	--warmup-updates 1000 \
	--restore-file checkpoints/ar_2_finetuned/checkpoint_last.pt
}


# Train the free parameters for language 2 i.e. hebrew in this case, for 20 epochs
# --dataset-name = he, --dataset-idx = 3. use flag --finetune

function fine_he {
    MAX_EPOCH=90 # Also change this line
    python train.py config.yaml \
    --task shared-multilingual-translation \
    --dataset-name he \
    --dataset-idx 3 \
    --bilingual \
    --finetune \
    --max-epoch $MAX_EPOCH \
    --num-workers 0 \
    --arch transformer \
    --max-tokens $MAX_TOKENS \
    --lr $LR \
    --min-lr 1e-9 \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --save-dir $CHECKPOINTS \
    --log-format simple \
    --log-interval 10 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --dropout 0.3 --weight-decay 0.0001 \
    --attention-dropout 0.1 \
    --activation-dropout 0.1 \
    --ddp-backend no_c10d \
    --update-freq $UPDATE_FREQ \
    --share-all-embeddings \
    --lr-scheduler inverse_sqrt \
    --warmup-init-lr 1e-07 \
    --warmup-updates 1000 \
    --reset-optimizer \
    --reset-lr-scheduler \
    --restore-file $CHECKPOINTS/ar_2_final/checkpoint_last.pt
}


#Prune the parameters of the current language pair only and retrain it for 10 epochs to compensate for heavy pruning
# --dataset-name = he, --dataset-idx = 3, use flag --prune
function prune_he {
	MAX_EPOCH=100
	python train.py config.yaml \
	--task shared-multilingual-translation \
	--dataset-name he \
	--dataset-idx 3 \
	--bilingual \
	--prune \
	--prune-perc 0.75 \
	--max-epoch $MAX_EPOCH \
	--num-workers 0 \
	--arch transformer \
	--max-tokens $MAX_TOKENS \
	--lr $LR \
	--min-lr 1e-9 \
	--optimizer adam --adam-betas '(0.9, 0.98)' \
	--save-dir $CHECKPOINTS \
	--log-format simple \
	--log-interval 10 \
	--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
	--dropout 0.3 --weight-decay 0.0001 \
	--attention-dropout 0.1 \
	--activation-dropout 0.1 \
	--ddp-backend no_c10d \
	--update-freq $UPDATE_FREQ \
	--share-all-embeddings \
	--lr-scheduler inverse_sqrt \
	--warmup-init-lr 1e-07 \
	--warmup-updates 1000 \
	--restore-file checkpoints/he_3_finetuned/checkpoint_last.pt
}

# Change warmup-updates and save-interval-updates according to number of batches
train_MNMT
prune_MNMT
fine_ar
prune_ar
fine_he