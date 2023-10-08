import pandas as pd
import numpy as np
from util.TextPreprocessing import clean_text

codes = ['PF000', 'PD000', 'PE000', 'PMG00', 'PMP00', 'PMI00', 'PMA00', 'PME00', 'PMC00', 'PMN00', 'PMR00', 'PMO00', 'PA000', 'PO000', 'PB000', 'PGI00', 'PC000', 'PG000', 'PS000', 'PH000', 'PV000', 'PU000', 'PY000', 'PL000', 'PN000', 'PR000', 'PP000', 'PJ000', 'PT000', 'PX000', 'PK000', 'PQL00', 'PQ000']

def split_dataframe(df, ratios):
    # 비율 합이 1인지 확인
    if sum(ratios) != 1:
        raise ValueError("비율의 합은 1이어야 합니다.")

    # 데이터프레임을 세 가지 부분으로 나눔
    start_idx = 0
    file_name = ["SFT", "RM", "PPO"]
    for i, ratio in enumerate(ratios):
        end_idx = start_idx + int(len(df) * ratio)
        tmp_df = df.iloc[start_idx:end_idx]
        tmp_df.to_csv(f'../RL/data/original/{file_name[i]}.csv', index=False)
        start_idx = end_idx

def search_random_answer(code, df):
    current_code = code
    tmp_code_list = [code for code in code_list if current_code != code]
    random_code = np.random.choice(tmp_code_list)
    random_rows = df[df['code'] == random_code]
    random_row = random_rows.sample(n=1, ignore_index=True)
    r_code = random_row['code'].values
    r_cleaned_answer = random_row['cleaned_answer'].values
    r_refer = random_row['url'].values
    return pd.Series({'r_code': r_code[0], 'r_cleaned_answer': r_cleaned_answer[0], 'r_refer': r_refer[0]})

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
        (nonnullDf['cleaned_question_len'] >= 50) &
        (nonnullDf['cleaned_question_len'] <= 300) &
        (nonnullDf['cleaned_answer_len'] >= 30) &
        (nonnullDf['cleaned_answer_len'] <= 300)
    ]
    lengthDf = lengthDf.sample(frac=1)
    print(f'길이 제거 후 데이터 : {len(lengthDf)}')

    # SFT, RM, PPO data ratio
    ratios = [0.4, 0.3, 0.3]
    split_dataframe(lengthDf, ratios)

    # RM dataset
    rm_df = pd.read_csv('../RL/data/original/RM.csv')
    code_list = list(set(rm_df['code'].values))
    # rm_df[['r_code', 'r_cleaned_answer', 'r_refer']] = rm_df['code'].apply(search_random_answer)
    rm_df[['r_code', 'r_cleaned_answer', 'r_refer']] = rm_df.apply(lambda row: search_random_answer(row['code'], rm_df), axis=1)
    rm_df.to_csv('../RL/data/RM/RM_dataset.csv', index=False)





