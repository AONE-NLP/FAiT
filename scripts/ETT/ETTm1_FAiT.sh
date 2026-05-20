if [ ! -d "./logs/freformer/ablation/attn" ]; then
    mkdir -p ./logs/freformer/ablation/attn
fi
# forecasting_ETTm1_freformer_96_96_e1_s0_p1_o1_attn_ablation_250826_173455_FAiT_ETTm1_sl-96_pl-96_var-7_dm-1024_stages-4_2P4i_1P1i_dec-2_des-Exp_0  
# mae: 0.33932, mse: 0.30512, r2: -20.98860, pear: 0.61274, mase: 4.09838

# forecasting_ETTm1_FAiT_96_192_d1024_e32_b8_l4_250905_142713_FAiT_ETTm1_sl-96_pl-192_var-7_dm-1024_stages-4_2P4i_1P1i_dec-2_des-Exp_0  
# mae: 0.36751, mse: 0.36047, r2: -0.20660, pear: 0.57545, mase: 3.46561

# forecasting_ETTm1_freformer_96_336_e1_s0_p1_o1_attn_ablation_250826_195305_FAiT_ETTm1_sl-96_pl-336_var-7_dm-1024_stages-4_2P4i_1P1i_dec-2_des-Exp_0  
# mae: 0.38828, mse: 0.39020, r2: -0.16055, pear: 0.54462, mase: 3.65748

# forecasting_ETTm1_freformer_96_720_e1_s0_p1_o1_attn_ablation_250826_210205_FAiT_ETTm1_sl-96_pl-720_var-7_dm-1024_stages-4_2P4i_1P1i_dec-2_des-Exp_0  
# mae: 0.42548, mse: 0.45635, r2: -0.13203, pear: 0.50574, mase: 4.01669

model_name=FAiT


seq_lens=(96 96 96 96)
pred_lens=(96 192 336 720)
train_ratios=(1.0 1.0 1.0 1.0)

d_models=(1024 1024 1024 1024)
embed_size=(16 32 16 16)
bs=(32 8 64 32)
num_layer=(2 4 2 2)

cuda_ids1=(0 0 0 0)
learning_rate=(1e-4 1e-4 1e-4 1e-4)
dropout=(0.1 0.1 0.1 0.1)

attn_enhance=1
attn_softmax_flag=0
attn_weight_plus=1
attn_outside_softmax=1


for ((i = 0; i < 4; i++))
do

    seq_len=${seq_lens[i]}
    pred_len=${pred_lens[i]}

    export CUDA_VISIBLE_DEVICES=${cuda_ids1[i]}

    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTm1.csv \
      --model_id ETTm1_freformer_${seq_len}_${pred_len}_e${attn_enhance}_s${attn_softmax_flag}\
_p${attn_weight_plus}_o${attn_outside_softmax}_attn_ablation \
      --model $model_name \
      --data ETTm1 \
      --features M \
      --seq_len ${seq_len} \
      --pred_len ${pred_len} \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --embed_size ${embed_size[i]} \
      --d_model ${d_models[i]} \
      --d_ff ${d_models[i]} \
      --batch_size 32 \
      --learning_rate ${learning_rate[i]} \
      --itr 1 \
      --e_layers 2 \
      --lossfun_alpha 0.5 \
      --test_batch_size 16 \
      --test_mode 0 \
      --CKA_flag 0 \
      --fix_seed 1 \
      --resume_training 0 \
      --resume_epoch 0 \
      --save_every_epoch 0 \
      --use_revin 1 \
      --use_norm 1 \
      --send_mail 0 \
      --save_pdf 0 \
      --train_epochs 30 \
      --patience 4 \
      --lradj type1 \
      --loss_mode L1 \
      --train_ratio 1.0 \
      --dropout ${dropout[i]} \
      --plot_mat_flag 1 \
      --attn_enhance ${attn_enhance} \
      --attn_softmax_flag ${attn_softmax_flag} \
      --attn_weight_plus ${attn_weight_plus} \
      --attn_outside_softmax ${attn_outside_softmax} \
      2>&1 | tee -a logs/freformer/ablation/attn/Freformer_attn_abl_ETTm1_${seq_len}_${pred_len}_e${attn_enhance}\
_s${attn_softmax_flag}_p${attn_weight_plus}_o${attn_outside_softmax}.log 
done

