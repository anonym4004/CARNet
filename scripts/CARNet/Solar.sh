
model_name=carnet
root_path_name=C:/Users/Awsftausif/Desktop/S-Mamba_datasets/Solar/
data_path_name=solar_AL.txt
model_id_name=Solar
data_name=Solar
random_seed=2024

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
  --pred_len 96 \
  --e_layers 3 \
  --enc_in 137 \
  --n_vars 137 \
  --d_model 512 \
  --batch_size 16 \
  --d_ff 512 \
  --d_core 128 \
  --learning_rate 0.0001 \
  --cycle 144 \
  --des 'Exp' \
  --use_norm 0 \
  --freq t \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_96_192' \
  --model $model_name \
  --data $data_name \
  --features M \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --enc_in 137 \
  --n_vars 137 \
  --d_model 512 \
  --batch_size 16 \
  --d_ff 512 \
  --d_core 128 \
  --learning_rate 0.0001 \
  --cycle 144 \
  --des 'Exp' \
  --use_norm 0 \
  --freq t \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_96_336' \
  --model $model_name \
  --data $data_name \
  --features M \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --enc_in 137 \
  --n_vars 137 \
  --d_model 512 \
  --batch_size 16 \
  --d_ff 512 \
  --d_core 128 \
  --learning_rate 0.0001 \
  --cycle 144 \
  --des 'Exp' \
  --use_norm 0 \
  --freq t \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_96_720' \
  --model $model_name \
  --data $data_name \
  --features M \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720\
  --e_layers 3 \
  --enc_in 137 \
  --n_vars 137 \
  --d_model 512 \
  --batch_size 16 \
  --d_ff 512 \
  --d_core 128 \
  --learning_rate 0.0001 \
  --cycle 144 \
  --des 'Exp' \
  --use_norm 0 \
  --freq t \
  --itr 1