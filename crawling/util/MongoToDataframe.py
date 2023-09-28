import pandas as pd
from crawling.database.Mongodb_connect import MongodbConnection

# 몽고 디비에 저장된 데이터를 데이터 프레임으로 변환

codes = ['PF000', 'PD000', 'PE000', 'PMG00', 'PMP00', 'PMI00', 'PMA00', 'PME00', 'PMC00', 'PMN00', 'PMR00', 'PMO00', 'PA000', 'PO000', 'PB000', 'PGI00', 'PC000', 'PG000', 'PS000', 'PH000', 'PV000', 'PU000', 'PY000', 'PL000', 'PN000', 'PR000', 'PP000', 'PJ000', 'PT000', 'PX000', 'PK000', 'PQL00', 'PQ000']
save_dir = "../data/"

if __name__ == '__main__':
    mongo = MongodbConnection()
    # results = mongo.find_all(collection='doctor-qa-with-rl')
    results = mongo.find(collection='doctor-qa-with-rl', filter={"code":"PA000"})
    rows = []
    for result in results:
        for answer in result['answer']:
            rowdict = {}
            rowdict['id'] = result['_id']
            rowdict['question'] = result['question']
            rowdict['answer'] = answer['answer']
            rowdict['hiddenAnswerName'] = answer['hiddenAnswerName']
            rowdict['hiddenRegMemberUid'] = answer['hiddenRegMemberUid']
            rowdict['hiddenScore'] = answer['hiddenScore']
            rowdict['counselAnswerCid'] = answer['counselAnswerCid']
            rowdict['viewId'] = result['viewId']
            rowdict['url'] = result['url']
            rows.append(rowdict)

    df = pd.DataFrame(rows)
    df.to_csv(save_dir + "original_sample.csv", sep='\t', index=False)
