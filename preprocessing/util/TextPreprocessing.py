import re
import logging
import numpy as np
def clean_text(text):
    try:
        # 안녕하세요 누구누구 입니다 제거
        pattern = '(안녕하세요).+?(입니다.)'
        text = re.sub(pattern=pattern, repl=' ', string=text)

        # 삭제된 질문 제거
        pattern = re.compile(r'본 게시물은 질문자 본인의 요청으로 삭제되었습니다.')
        text = re.sub(pattern, "", text)

        # 감사합니다 제거
        pattern = r"감사합니다\.$|감사합니다$"
        text = re.sub(pattern, "", text)

        # 이메일 제거
        pattern = '([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)'
        text = re.sub(pattern=pattern, repl='', string=text)

        # 전화번호 제거
        pattern = '/(\d{3}).*(\d{3}).*(\d{4})/'
        text = re.sub(pattern=pattern, repl=' ', string=text)

        #  URL 제거
        pattern = '(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
        text = re.sub(pattern=pattern, repl=' ', string=text)

        # 한글 자음 모음 제거
        pattern = '([ㄱ-ㅎㅏ-ㅣ]+)'
        text = re.sub(pattern=pattern, repl=' ', string=text)

        # html 태그 제거
        pattern = '<[^>]*>'
        text = re.sub(pattern=pattern, repl='', string=text)

        # 기타 문자 제거
        text = re.sub(r'&lt;o:p&gt;.*?&lt;/o:p&gt;', '', text)

        # 괄호 사이 글자 제거
        pattern = r'\([^)]*\)'
        text = re.sub(pattern=pattern, repl=' ', string=text)

        # &nbsp; 제거
        text = re.sub(r'\xa0', ' ', text)

        # \n\n -> \n
        text = re.sub(r'\n+', '\n', text)

        # 연속된 특수 문자 한개로 만들기
        pattern = r'([^\w\s])\1+'
        text = re.sub(pattern=pattern, repl=r'\1', string=text)

        # 양 끝 공백 제거
        text = text.strip()




    except TypeError as e:
        logging.info(f'error occurred : {text} , {e}')

    return text