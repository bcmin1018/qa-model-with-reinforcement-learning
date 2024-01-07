## Application of Reinforcement Learning to Response Generation Model Using Domain Data
+ 크롤링한 데이터를 ChatGPT 방식을 적용하여 학습
+ HumanFeedback 대신 지표를 사용해서 응답을 차등화한 보상 모델 학습

1. Supervised fine-tuning of the base llama-7b model
python /app/SFT_trl.py --mode train --train_path /app/input/dataset/sft/SFT.csv --num_train_epochs 20 --per_device_train_batch_size 32 --per_device_eval_batch_size 16 --learning_rate 1e-5 --model_name gpt3 --save_steps 1000 --logging_steps 300 --eval_steps 1000 --lr_scheduler_type cosine --hub_model_id bradmin/sft_trl_adapter_200 --seq_length 512
