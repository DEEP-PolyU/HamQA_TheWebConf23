export CUDA_VISIBLE_DEVICES=0
dt=`date '+%Y%m%d_%H%M%S'`

dataset="obqa"
model='roberta-large'
shift
shift
args=$@


echo "***** hyperparameters *****"
echo "dataset: $dataset"
echo "******************************"

save_dir_pref='saved_models'
mkdir -p $save_dir_pref

###### Training ######

python3 -u HamQA.py --dataset $dataset --mode eval_detail \
    --train_adj data/${dataset}/graph/train.graph.adj.pk \
    --dev_adj   data/${dataset}/graph/dev.graph.adj.pk \
    --test_adj  data/${dataset}/graph/test.graph.adj.pk \
    --train_statements  data/${dataset}/statement/train.statement.jsonl \
    --dev_statements  data/${dataset}/statement/dev.statement.jsonl \
    --test_statements  data/${dataset}/statement/test.statement.jsonl \
    --load_model_path ${save_dir_pref}/obqa_model.pt \
    $args

