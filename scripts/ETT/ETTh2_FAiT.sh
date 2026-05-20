if [ ! -d "./logs/FAiT/ablation/attn" ]; then
    mkdir -p ./logs/FAiT/ablation/attn
fi

# forecasting_ETTh2_FAiT_96_96_e1_s0_p1_o1_attn_ablation_250826_174132_FAiT_ETTh2_sl-96_pl-96_var-7_dm-1024_stages-4_2P1i_1P1i_dec-2_des-Exp_0  
# mae: 0.33092, mse: 0.28089, r2: -1.20227, pear: 0.42591, mase: 4.43593

# forecasting_ETTh2_freformer_96_192_e1_s0_p1_o1_attn_ablation_250826_175234_FAiT_ETTh2_sl-96_pl-192_var-7_dm-1024_stages-4_2P1i_1P1i_dec-2_des-Exp_0  
# mae: 0.38097, mse: 0.36213, r2: -0.59333, pear: 0.38307, mase: 3.20924

# forecasting_ETTh2_d1024_e8_b64_l2_250827_020036_FAiT_ETTh2_sl-96_pl-336_var-7_dm-1024_stages-4_2P2i_1P1i_dec-2_des-Exp_0  
# mae: 0.41546, mse: 0.40480, r2: -0.44779, pear: 0.35148, mase: 3.17507

# forecasting_ETTh2_FAiT_96_720_e1_s0_p1_o1_attn_ablation_250826_181231_FAiT_ETTh2_sl-96_pl-720_var-7_dm-1024_stages-4_2P1i_1P1i_dec-2_des-Exp_0  
# mae: 0.43182, mse: 0.41247, r2: -0.31116, pear: 0.29772, mase: 3.15556

model_name=FAiT

seq_lens=(96 96 96 96)
pred_lens=(96 192 336 720)
train_ratios=(1.0 1.0 1.0 1.0)

d_models=(1024 1024 1024 1024)  # new
embed_size=(8 8 8 8)
bs=(32 32 64 32)
num_layer=(2 2 2 2)

cuda_ids1=(0 0 0 0)

learning_rate=(1e-4 1e-4 1e-4 1e-4)
dropout=(0.2 0.2 0.2 0.2)

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
      --data_path ETTh2.csv \
      --model_id ETTh2_FAiT_${seq_len}_${pred_len}_e${attn_enhance}_s${attn_softmax_flag}\
_p${attn_weight_plus}_o${attn_outside_softmax}_attn_ablation \
      --model $model_name \
      --data ETTh2 \
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
      --batch_size ${bs[i]} \
      --learning_rate ${learning_rate[i]} \
      --itr 1 \
      --e_layers ${num_layer[i]} \
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
      --plot_mat_flag 0 \
      --attn_enhance ${attn_enhance} \
      --attn_softmax_flag ${attn_softmax_flag} \
      --attn_weight_plus ${attn_weight_plus} \
      --attn_outside_softmax ${attn_outside_softmax} \
      2>&1 | tee -a logs/FAiT/ablation/attn/FAiT_attn_abl_ETTh2_${seq_len}_${pred_len}_e${attn_enhance}\
_s${attn_softmax_flag}_p${attn_weight_plus}_o${attn_outside_softmax}.log 
done
