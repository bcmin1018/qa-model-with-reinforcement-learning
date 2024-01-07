## Application of Reinforcement Learning to Response Generation Model Using Domain Data
+ 크롤링한 데이터로 polyglot-ko-1.3b 모델을 ChatGPT 학습 방식으로 파인튜닝 및 강화 학습 진행
+ HumanFeedback 대신 지표를 사용해서 응답을 차등화한 보상 모델 학습
+ 논문 링크 : 

1. Supervised fine-tuning(SFT) of the base EleutherAI/polyglot-ko-1.3b
```
python SFT_trl.py --mode train --train_path {TRAIN_PATH} --num_train_epochs 20 --per_device_train_batch_size 32 --per_device_eval_batch_size 16 --learning_rate 1e-5 --model_name gpt3 --save_steps 1000 --logging_steps 300 --eval_steps 1000 --lr_scheduler_type cosine --seq_length 512
```

2. Reward modeling(RM) using QA data pairs with 1.SFT model
```
python RM.py --model_name=klue/roberta-large --tokenizer_name=klue/roberta-large --per_device_train_batch_size=6 --per_device_eval_batch_size=6 --num_labels=1 --train_path={TRAIN_PATH} --learning_rate=9e-5 --lr_scheduler_type=cosine --gradient_accumulation_steps=10 --logging_steps=10 --eval_steps 100 --save_steps 100 --num_train_epochs=1
```

3. RL fine-tuning with 1.SFT Model and 2.RM Model
```
python PPO.py --mode train --train_path {TRAIN_PATH} --batch_size 64 --save_freq 50 --learning_rate 9e-7 --ppo_epochs 5 --mini_batch_size 16 --gradient_accumulation_steps 4 --adap_kl_ctrl true --early_stopping True --tracker_project_name ppo --model_name {SFT_PATH} --reward_model_name {RM_PATH} --tokenizer EleutherAI/polyglot-ko-1.3b --reward_tokenizer EleutherAI/polyglot-ko-1.3b --kl_penalty full --score_clip 0.6
```
