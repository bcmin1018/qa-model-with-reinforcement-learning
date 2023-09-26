from crawling.database.Mongodb_connect import MongodbConnection
import pandas as pd

codes = ['PF000', 'PD000', 'PE000', 'PMG00', 'PMP00', 'PMI00', 'PMA00', 'PME00', 'PMC00', 'PMN00', 'PMR00', 'PMO00', 'PA000', 'PO000', 'PB000', 'PGI00', 'PC000', 'PG000', 'PS000', 'PH000', 'PV000', 'PU000', 'PY000', 'PL000', 'PN000', 'PR000', 'PP000', 'PJ000', 'PT000', 'PX000', 'PK000', 'PQL00', 'PQ000']

if __name__ == '__main__':
    mongo = MongodbConnection()
    result = mongo.find_all(collection='doctor-qa-with-rl')
    df = pd.DataFrame(list(result))

