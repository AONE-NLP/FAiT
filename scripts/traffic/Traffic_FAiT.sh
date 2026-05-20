if [ ! -d "./logs/FAiT/ablation/attn" ]; then
    mkdir -p ./logs/FAiT/ablation/attn
fi

model_name=FAiT

# forecasting_Traffic_d512_e16_b32_l2_lr5e-4_250828_030120_FAiT_custom_sl-96_pl-96_var-862_dm-512_stages-4_2P2i_1P1i_dec-2_des-Exp_0  
# mae: 0.24479, mse: 0.39936, r2: 0.71062, pear: 0.89005, mase: 0.79416

# forecasting_Traffic_d512_e32_b16_l2_lr5e-4_250903_213134_FAiT_custom_sl-96_pl-192_var-862_dm-512_stages-4_2P2i_1P1i_dec-2_des-Exp_0  
# mae: 0.25296, mse: 0.42557, r2: 0.69415, pear: 0.87695, mase: 0.80412

# forecasting_Traffic_d512_e32_b16_l3_lr5e-4_250901_194655_FAiT_custom_sl-96_pl-336_var-862_dm-512_stages-4_2P3i_1P1i_dec-2_des-Exp_0  
# mae: 0.25720, mse: 0.43746, r2: 0.71664, pear: 0.86931, mase: 0.78651

# forecasting_Traffic_d128_e32_b8_l3_lr5e-4_250906_013223_FAiT_custom_sl-96_pl-720_var-862_dm-128_stages-4_2P3i_1P1i_dec-2_des-Exp_0  
# mae: 0.27928, mse: 0.48309, r2: 0.69098, pear: 0.85375, mase: 0.84776

seq_lens=(96 96 96 96)
pred_lens=(96 192 336 720)
train_ratios=(1.0 1.0 1.0 1.0)

d_models=(512 512 512 128)
embed_size=(16 32 32 32)
bs=(32 16 16 8)
num_layer=(2 2 3 3)


cuda_ids1=(1 1 1 1)

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
      --root_path ./dataset/traffic/ \
      --data_path traffic.csv \
      --model_id Traffic_FAiT_${seq_len}_${pred_len}_e${attn_enhance}_s${attn_softmax_flag}\
_p${attn_weight_plus}_o${attn_outside_softmax}_attn_ablation \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len ${seq_len} \
      --pred_len ${pred_len} \
      --enc_in 862 \
      --dec_in 862 \
      --c_out 862 \
      --des 'Exp' \
      --embed_size ${embed_size[i]} \
      --d_model ${d_models[i]} \
      --d_ff ${d_models[i]} \
      --batch_size ${bs[i]} \
      --learning_rate 5e-4 \
      --itr 1 \
      --e_layers ${num_layer[i]} \
      --lossfun_alpha 0.5 \
      --test_batch_size 16 \
      --test_mode 2 \
      --CKA_flag 0 \
      --fix_seed 1 \
      --resume_training 0 \
      --resume_epoch 27 \
      --save_every_epoch 0 \
      --use_revin 1 \
      --use_norm 1 \
      --send_mail 0 \
      --save_pdf 1 \
      --train_epochs 50 \
      --patience 4 \
      --lradj cosine \
      --loss_mode L1 \
      --train_ratio 1.0 \
      --dropout 0.0 \
      --plot_mat_flag 0 \
      --attn_enhance ${attn_enhance} \
      --attn_softmax_flag ${attn_softmax_flag} \
      --attn_weight_plus ${attn_weight_plus} \
      --attn_outside_softmax ${attn_outside_softmax} \
      2>&1 | tee -a logs/FAiT/ablation/attn/FAiT_attn_abl_Traffic_${seq_len}_${pred_len}_e${attn_enhance}\
_s${attn_softmax_flag}_p${attn_weight_plus}_o${attn_outside_softmax}.log
done