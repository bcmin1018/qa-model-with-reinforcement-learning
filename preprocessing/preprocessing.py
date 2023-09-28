import pandas as pd
from util.TextPreprocessing import clean_text

codes = ['PF000', 'PD000', 'PE000', 'PMG00', 'PMP00', 'PMI00', 'PMA00', 'PME00', 'PMC00', 'PMN00', 'PMR00', 'PMO00', 'PA000', 'PO000', 'PB000', 'PGI00', 'PC000', 'PG000', 'PS000', 'PH000', 'PV000', 'PU000', 'PY000', 'PL000', 'PN000', 'PR000', 'PP000', 'PJ000', 'PT000', 'PX000', 'PK000', 'PQL00', 'PQ000']

if __name__ == '__main__':
    originalDf = pd.read_csv('/crawling/data/original_sample.csv', sep='\t')
    print(f'원본 데이터 길이: {len(originalDf)}')

    nonduplicateDf = originalDf.drop_duplicates(subset='counselAnswerCid')
    print(f'중복된 행 제거 후 데이터 길이: {len(nonduplicateDf)}')

    nonduplicateDf['cleaned_question'] = nonduplicateDf['question'].apply(clean_text)
    nonduplicateDf['clenaed_answer'] = nonduplicateDf['answer'].apply(clean_text)
    nonduplicateDf['clenaed_question_len'] = nonduplicateDf['cleaned_question'].astype('str').apply(len)
    nonduplicateDf['clenaed_answer_len'] = nonduplicateDf['clenaed_answer'].astype('str').apply(len)

    nonnullDf = nonduplicateDf.dropna(axis=0)
    print(f'결측치 제거 후 데이터 길이: {len(nonnullDf)}')

    # 질문 / 답변 30자 이하 제거
    lengthDf = nonnullDf[(nonnullDf['clenaed_question_len'] >= 50) & (nonnullDf['clenaed_answer_len'] >= 30)]
    print(f'길이 제거 후 데이터 : {len(lengthDf)}')
    # 자리수 초과 제거

    lengthDf.to_csv('./data/afterPre.csv', index=False)


    def split_dataframe(df, ratios):
        """
        데이터프레임을 세 가지 비율로 나눠서 저장하는 함수

        Parameters:
            - df: 나눌 데이터프레임
            - ratios: 세 가지 비율을 담은 리스트, 비율의 합은 1이어야 함

        Returns:
            - divided_dfs: 비율에 따라 나눠진 데이터프레임들을 담은 딕셔너리
        """
        # 비율 합이 1인지 확인
        if sum(ratios) != 1:
            raise ValueError("비율의 합은 1이어야 합니다.")

        # 데이터프레임을 세 가지 부분으로 나눔
        num_parts = len(ratios)
        start_idx = 0
        divided_dfs = {}
        for i, ratio in enumerate(ratios):
            end_idx = start_idx + int(len(df) * ratio)
            divided_dfs[f'part_{i + 1}'] = df.iloc[start_idx:end_idx]
            start_idx = end_idx

        return divided_dfs


    # 사용 예시
    if __name__ == "__main__":
        # 데이터프레임 생성 예시
        data = {
            'A': [1, 2, 3, 4, 5],
            'B': [6, 7, 8, 9, 10]
        }
        df = pd.DataFrame(data)

        # 비율 설정 (예: 0.4, 0.3, 0.3)
        ratios = [0.4, 0.3, 0.3]

        # 데이터프레임을 나누고 저장
        divided_dfs = split_dataframe(df, ratios)

        # 결과 출력
        for key, divided_df in divided_dfs.items():
            print(f"{key} 데이터프레임:")
            print(divided_df)
            print("\n")





