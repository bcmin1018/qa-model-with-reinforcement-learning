import logging
import math
import random
import time
from datetime import datetime
from pytz import timezone
datetime.now(timezone('Asia/Seoul'))
import requests
from bs4 import BeautifulSoup
import re



class HidocCrawling():
    def __init__(self, url):
        self.main_url = url

    # 모든 진료 과목 코드를 조회 하는 함수
    def findMedicalSpecialtyByClassName(self, class_name):
        '''
        :param class_name: html에 있는 class name
        :return: code list
        '''
        raw = requests.get(self.main_url, headers={'User-Agent': 'Mozilla/5.0'})
        time.sleep(random.uniform(1,3))
        html = BeautifulSoup(raw.text, "html.parser")
        Elements = html.find_all(class_=class_name)
        code = [re.findall("(?<=code=).+(?=\")", str(e))[0] for e in Elements]
        return code

    # 상담 페이지 ID 수집
    def getViewIds(self, code, page):
        headers = {'User-Agent': 'Mozilla/5.0'}
        params = {
            "code" : code,
            "page" : page
        }
        raw = requests.get(self.main_url, headers=headers, params=params)
        html = BeautifulSoup(raw.text, "html.parser")
        Elements = html.find_all("div", class_="box_type1 qna_main")
        viewIds = []
        for e in Elements:
            a_tag = e.find("a")
            viewId = a_tag.get("href").split('/')[1]
            viewIds.append(viewId)
        return viewIds

    # 진료 과목 총 페이지 수 조
    def getPageNum(self, code):
        headers = {'User-Agent': 'Mozilla/5.0'}
        params = {
            "code" : code
        }
        raw = requests.get(self.main_url, headers=headers, params=params)
        html = BeautifulSoup(raw.text, "html.parser")
        Elements = html.find("div", class_="tit_summary")
        total = ''.join(re.findall("\d", str(Elements)))
        pageNum = int(total) / 7
        return math.trunc(pageNum)

    def getQatext(self, viewId):
        headers = {'User-Agent': 'Mozilla/5.0'}
        raw = requests.get("https://www.hidoc.co.kr/healthqna/part" + "/view/" + viewId, headers=headers)
        html = BeautifulSoup(raw.text, "html.parser")

        # 질문
        question = html.find('div', {'class': 'box_type1 view_question'}).find('p')
        question = ''.join([str(item).replace('<br/>', '\n') for item in question])

        # 답변
        answer_bodys = html.find_all('div', class_='answer_body')
        answer_list = []

        for answer_body in answer_bodys:
            try:
                answer = answer_body.find("div", class_="desc").get_text()
                hiddenAnswerName = answer_body.find(attrs={'class': 'hiddenAnswerName'}).attrs['value']  # 전문의명
                hiddenRegMemberUid = answer_body.find(attrs={'class': 'hiddenRegMemberUid'}).attrs['value']  # 전문의ID
                hiddenScore = answer_body.find(attrs={'class': 'hiddenScore'}).attrs['value']  # 전문의 점수
                counselAnswerCid = answer_body.find(attrs={'name': 'counselAnswerCid'}).attrs['value']  # 답

                answer_dict = {}
                answer_dict['answer'] = answer
                answer_dict['hiddenAnswerName'] = hiddenAnswerName
                answer_dict['hiddenRegMemberUid'] = hiddenRegMemberUid
                answer_dict['hiddenScore'] = hiddenScore
                answer_dict['counselAnswerCid'] = counselAnswerCid

                answer_list.append(answer_dict)
            except AttributeError as e:
                logging.info(f'{viewId} get an error : {e}')
                pass


        return question, answer_list, viewId