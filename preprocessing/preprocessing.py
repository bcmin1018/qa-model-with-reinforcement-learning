import pandas as pd
from util.TextPreprocessing import clean_text

codes = ['PF000', 'PD000', 'PE000', 'PMG00', 'PMP00', 'PMI00', 'PMA00', 'PME00', 'PMC00', 'PMN00', 'PMR00', 'PMO00', 'PA000', 'PO000', 'PB000', 'PGI00', 'PC000', 'PG000', 'PS000', 'PH000', 'PV000', 'PU000', 'PY000', 'PL000', 'PN000', 'PR000', 'PP000', 'PJ000', 'PT000', 'PX000', 'PK000', 'PQL00', 'PQ000']

def split_dataframe(df, ratios):
    # 비율 합이 1인지 확인
    if sum(ratios) != 1:
        raise ValueError("비율의 합은 1이어야 합니다.")

    # 데이터프레임을 세 가지 부분으로 나눔
    num_parts = len(ratios)
    start_idx = 0
    divided_dfs = {}
    file_name = ["SFT", "RM", "PPO"]
    for i, ratio in enumerate(ratios):
        end_idx = start_idx + int(len(df) * ratio)
        df = df.iloc[start_idx:end_idx]
        df.to_csv(f'../RL/data/original/{file_name[i]}.csv', index=False)
        start_idx = end_idx

    return divided_dfs


if __name__ == '__main__':
    originalDf = pd.read_csv('../crawling/data/original_sample.csv', sep='\t')
    print(f'원본 데이터 길이: {len(originalDf)}')

    nonduplicateDf = originalDf.drop_duplicates(subset='counselAnswerCid')
    print(f'중복된 행 제거 후 데이터 길이: {len(nonduplicateDf)}')

    nonduplicateDf['cleaned_question'] = nonduplicateDf['question'].apply(clean_text)
    nonduplicateDf['clenaed_answer'] = nonduplicateDf['answer'].apply(clean_text)
    nonduplicateDf['clenaed_question_len'] = nonduplicateDf['cleaned_question'].astype('str').apply(len)
    nonduplicateDf['clenaed_answer_len'] = nonduplicateDf['clenaed_answer'].astype('str').apply(len)

    nonnullDf = nonduplicateDf.dropna(axis=0)
    print(f'결측치 제거 후 데이터 길이: {len(nonnullDf)}')

    # 질문 / 답변 30자 이하 제거 300자 이상 제거
    lengthDf = nonnullDf[
        (nonnullDf['clenaed_question_len'] >= 50) &
        (nonnullDf['clenaed_question_len'] <= 300) &
        (nonnullDf['clenaed_answer_len'] >= 30) &
        (nonnullDf['clenaed_answer_len'] <= 300)
    ]
    print(f'길이 제거 후 데이터 : {len(lengthDf)}')

    # SFT, RM, PPO data ratio
    ratios = [0.4, 0.3, 0.3]

    divided_dfs = split_dataframe(lengthDf, ratios)
    for key, divided_df in divided_dfs.items():
        print(f"{key} 데이터프레임:")
        print(divided_df)
        print("\n")

    lengthDf.to_csv('./data/afterPre.csv', index=False)






