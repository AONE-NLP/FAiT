if [ ! -d "./logs/freformer/ablation/attn" ]; then
    mkdir -p ./logs/freformer/ablation/attn
fi

# forecasting_ETTm2_d1024_e32_b8_l1_250825_070516_FAiT_ETTm2_sl-96_pl-96_var-7_dm-1024_stages-4_2P1i_1P1i_dec-2_des-Exp_0  
# mae: 0.24844, mse: 0.17028, r2: -6.80100, pear: 0.45803, mase: 6.21609

# forecasting_ETTm2_d256_e16_b8_l2_250830_224420_FAiT_ETTm2_sl-96_pl-192_var-7_dm-256_stages-4_2P2i_1P1i_dec-2_des-Exp_0  
# mae: 0.28979, mse: 0.23340, r2: -1.33996, pear: 0.43098, mase: 6.62798

# forecasting_ETTm2_d1024_e8_b64_l1_250827_190345_FAiT_ETTm2_sl-96_pl-336_var-7_dm-1024_stages-4_2P1i_1P1i_dec-2_des-Exp_0  
# mae: 0.33034, mse: 0.29228, r2: -0.93702, pear: 0.37731, mase: 8.89269

# forecasting_ETTm2_d512_e32_b32_l1_250830_073216_FAiT_ETTm2_sl-96_pl-720_var-7_dm-512_stages-4_2P1i_1P1i_dec-2_des-Exp_0  
# mae: 0.38765, mse: 0.39080, r2: -0.67531, pear: 0.34809, mase: 7.71599

model_name=FAiT

seq_lens=(96 96 96 96)
pred_lens=(96 192 336 720)
train_ratios=(1.0 1.0 1.0 1.0)

d_models=(1024 256 1024 512)
embed_size=(32 16 8 32)
bs=(8 8 64 32)
num_layer=(1 2 1 1)

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
      --data_path ETTm2.csv \
      --model_id ETTm2_freformer_${seq_len}_${pred_len}_e${attn_enhance}_s${attn_softmax_flag}\
_p${attn_weight_plus}_o${attn_outside_softmax}_attn_ablation \
      --model $model_name \
      --data ETTm2 \
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
      --patience 3 \
      --lradj type1 \
      --loss_mode L1 \
      --train_ratio 1.0 \
      --dropout ${dropout[i]} \
      --plot_mat_flag 0 \
      --attn_enhance ${attn_enhance} \
      --attn_softmax_flag ${attn_softmax_flag} \
      --attn_weight_plus ${attn_weight_plus} \
      --attn_outside_softmax ${attn_outside_softmax} \
      2>&1 | tee -a logs/freformer/ablation/attn/Freformer_attn_abl_ETTm2_${seq_len}_${pred_len}_e${attn_enhance}\
_s${attn_softmax_flag}_p${attn_weight_plus}_o${attn_outside_softmax}.log 
done

