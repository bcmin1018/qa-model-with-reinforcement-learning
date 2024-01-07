import pandas as pd
import numpy as np
from util.TextPreprocessing import clean_text

codes = ['PF000', 'PD000', 'PE000', 'PMG00', 'PMP00', 'PMI00', 'PMA00', 'PME00', 'PMC00', 'PMN00', 'PMR00', 'PMO00', 'PA000', 'PO000', 'PB000', 'PGI00', 'PC000', 'PG000', 'PS000', 'PH000', 'PV000', 'PU000', 'PY000', 'PL000', 'PN000', 'PR000', 'PP000', 'PJ000', 'PT000', 'PX000', 'PK000', 'PQL00', 'PQ000']

def split_dataframe(df, ratios):
    start_idx = 0
    file_name = ["SFT", "RM", "PPO", "EVAL"]
    for i, ratio in enumerate(ratios):
        end_idx = start_idx + int(len(df) * ratio)
        tmp_df = df.iloc[start_idx:end_idx]
        print(f' {file_name[i]} 데이터 길이 : {len(tmp_df)}')
        tmp_df.to_csv(f'../RL/data/original/{file_name[i]}.csv', index=False)
        start_idx = end_idx

if __name__ == '__main__':
    originalDf = pd.read_csv('../crawling/data/original_sample.csv', sep='\t')
    print(f'원본 데이터 길이: {len(originalDf)}')

    nonduplicateDf = originalDf.drop_duplicates(subset='counselAnswerCid')
    print(f'중복된 행 제거 후 데이터 길이: {len(nonduplicateDf)}')

    nonduplicateDf['cleaned_question'] = nonduplicateDf['question'].apply(clean_text)
    nonduplicateDf['cleaned_answer'] = nonduplicateDf['answer'].apply(clean_text)
    nonduplicateDf['cleaned_question_len'] = nonduplicateDf['cleaned_question'].astype('str').apply(len)
    nonduplicateDf['cleaned_answer_len'] = nonduplicateDf['cleaned_answer'].astype('str').apply(len)

    nonnullDf = nonduplicateDf.dropna(axis=0)
    print(f'결측치 제거 후 데이터 길이: {len(nonnullDf)}')

    # 질문 / 답변 30자 이하 제거 300자 이상 제거
    lengthDf = nonnullDf[
        (nonnullDf['cleaned_question_len'] >= 30) &
        (nonnullDf['cleaned_question_len'] <= 300) &
        (nonnullDf['cleaned_answer_len'] >= 30) &
        (nonnullDf['cleaned_answer_len'] <= 300)
    ]
    lengthDf = lengthDf.sample(frac=1)
    print(f'길이 제거 후 데이터 : {len(lengthDf)}')

    # SFT, RM, PPO data ratio
    ratios = [0.2, 0.4, 0.3, 0.1]
    split_dataframe(lengthDf, ratios)

    # SFT 10% + PPO dataset
    sft_df = pd.read_csv('../RL/data/original/SFT.csv')
    ppo_df = pd.read_csv('../RL/data/original/PPO.csv')
    sample_size = int(0.1 * len(sft_df))
    sample_data = sft_df.sample(n=sample_size, random_state=2023)
    merged_data = pd.concat([ppo_df, sample_data])
    merged_data.to_csv('../RL/data/original/PPO.csv', index=False)
    print(f'새로운 PPO 데이터 길이 : {len(merged_data)}')