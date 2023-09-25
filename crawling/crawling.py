from database.Mongodb_connect import MongodbConnection
from util.HidocCrawling import HidocCrawling
import logging

logging.basicConfig(filename='crawler.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == '__main__':
    mongo = MongodbConnection()
    crawler = HidocCrawling(url="https://www.hidoc.co.kr/healthqna/part/list")
    codes = crawler.findMedicalSpecialtyByClassName('ico_comm link_view')

    code = input(f'무슨 진료 과목을 크롤링 한건가요? 진료 과목 코드 {codes}')
    totalPage = crawler.getPageNum(code)
    print(f'선택하신 {code}의 총 페이지는 {totalPage}입니다.')

    for page in range(369, totalPage):
        data = []

        print(f'현재 {page} 페이지 크롤링 시작')
        viewIds = crawler.getViewIds(code, page)
        for i, viewId in enumerate(viewIds):
            print(f'{i+1} / {len(viewIds)} {viewId} 진행 중...')
            question, answer = crawler.getQatext(viewId)

            insert_dict = {}
            insert_dict['code'] = code
            insert_dict['question'] = question
            insert_dict['answer'] = answer
            insert_dict['viewId'] = viewId
            insert_dict['url'] = url = "https://www.hidoc.co.kr/healthqna/part/view/" + viewId
            data.append(insert_dict)
        print(f'현재 {page} 페이지 크롤링 종료')
        mongo.insert_many('doctor-qa-with-rl', data)


