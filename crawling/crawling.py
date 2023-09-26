from database.Mongodb_connect import MongodbConnection
from util.HidocCrawling import HidocCrawling
from multiprocessing.dummy import Pool as ThreadPool
import logging

logging.basicConfig(filename='crawler.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


if __name__ == '__main__':
    mongo = MongodbConnection()
    crawler = HidocCrawling(url="https://www.hidoc.co.kr/healthqna/part/list")
    codes = crawler.findMedicalSpecialtyByClassName('ico_comm link_view')

    print(f'진료 과목 코드 {codes}')
    code = input(f'무슨 진료 과목을 크롤링 한건가요? : ')
    totalPage = crawler.getPageNum(code)
    logging.info(f'the total page of {code} is {totalPage}.')

    for page in range(1494, totalPage):
        viewIds = crawler.getViewIds(code, page)

        #멀티프로세스
        pool = ThreadPool(4)
        logging.info(f'current {page} page crawling start')
        results = pool.map(crawler.getQatext, viewIds)
        pool.close()
        pool.join()

        data = []
        for i, result in enumerate(results):
            question, answer, viewId = result
            logging.info(f'{i + 1} / {len(results)}  {viewId}  on progress...')

            insert_dict = {}
            insert_dict['code'] = code
            insert_dict['question'] = question
            insert_dict['answer'] = answer
            insert_dict['viewId'] = viewId
            insert_dict['url'] = url = "https://www.hidoc.co.kr/healthqna/part/view/" + viewId
            data.append(insert_dict)

        logging.info(f'current {page} page crawling stop')
        mongo.insert_many('doctor-qa-with-rl', data)

