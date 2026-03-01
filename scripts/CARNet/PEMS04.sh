
model_name=carnet
root_path_name=C:/Users/Awsftausif/Desktop/S-Mamba_datasets/PEMS/
data_path_name=PEMS04.npz
model_id_name=PEMS04
data_name=PEMS
random_seed=2024

python -u run.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_96_12' \
  --model $model_name \
  --data $data_name \
  --features M \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 12 \
  --e_layers 3 \
  --enc_in 307 \
  --n_vars 307 \
  --d_model 512 \
  --batch_size 16 \
  --d_ff 512 \
  --d_core 512 \
  --learning_rate 0.001 \
  --cycle 288 \
  --des 'Exp' \
  --freq t \
  --use_norm 0 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_96_24' \
  --model $model_name \
  --data $data_name \
  --features M \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 24 \
  --e_layers 3 \
  --enc_in 307 \
  --n_vars 307 \
  --d_model 512 \
  --batch_size 16 \
  --d_ff 512 \
  --d_core 512 \
  --learning_rate 0.001 \
  --cycle 288 \
  --des 'Exp' \
  --freq t \
  --use_norm 0 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_96_48' \
  --model $model_name \
  --data $data_name \
  --features M \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 48 \
  --e_layers 3 \
  --enc_in 307 \
  --n_vars 307 \
  --d_model 512 \
  --batch_size 16 \
  --d_ff 512 \
  --d_core 512 \
  --learning_rate 0.001 \
  --cycle 288 \
  --des 'Exp' \
  --freq t \
  --use_norm 0 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_96_96' \
  --model $model_name \
  --data $data_name \
  --features M \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96\
  --e_layers 3 \
  --enc_in 307 \
  --n_vars 307 \
  --d_model 512 \
  --batch_size 16 \
  --d_ff 512 \
  --d_core 512 \
  --learning_rate 0.001 \
  --cycle 288 \
  --des 'Exp' \
  --freq t \
  --use_norm 0 \
  --itr 1