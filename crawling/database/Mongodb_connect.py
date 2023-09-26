import pymongo
import os
# from crawling.util.JsonConfigFileManager import  JsonConfigFileManager
from util.JsonConfigFileManager import JsonConfigFileManager
from pathlib import Path


# https://www.bearpooh.com/171

root_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent

class MongodbConnection:
    def __init__(self):
        self.dbconfig = JsonConfigFileManager(os.path.join(root_dir, 'config/dbconfig.json'))
        self.user = self.dbconfig.values["username"]
        self.password = self.dbconfig.values["password"]
        self.host = self.dbconfig.values["host"]
        self.port = self.dbconfig.values["port"]
        self.database = self.dbconfig.values["database"]

        self.con = pymongo.MongoClient(
            "mongodb://{}:{}@{}:{}/{}".format(
                self.user,
                self.password,
                self.host,
                int(self.port),
                self.database)
        )
        self.db = self.con.get_database(self.dbconfig.values["database"])

    def insert_many(self, collection, data):
        self.db.get_collection(collection).insert_many(data)
        # self.db[collection].insert_many(data)

    def find(self, collection=None, filter=None, projection=None, skip=0, limit=0, sort=None, return_key=False):
        coll = self.db.get_collection(collection)
        results = coll.find(filter=filter, projection=projection, skip=skip, limit=limit, sort=sort, return_key=return_key)
        return results

    def find_all(self, collection):
        coll = self.db.get_collection(collection)
        result = coll.find({})
        documents = [document for document in result]
        return documents


    def find_distinct(self, collection=None, key=None, projection=None):
        '''
        :param collection:
        :param key:
        :param projection:  {col : 1, col2: 1}
        :return:
        '''
        result = self.db.get_collection(collection).aggregate([
    {"$group": {"_id": "$questionId", "question": {"$first": "$question"}, "answer": {"$first": "$answer"}, "code": {"$first": "$code"}, "url": {"$first": "$url"}}}
])
        # if projection is None:
        #     result = self.db.get_collection(collection).distinct(key)
        # else:
        #     result = self.db.get_collection(collection).distinct(key, projection)
        # distinct_col = self.db.get_collection(collection).distinct(key)
        # if key is None:
        #     query = {f'{key}' : {'$in': distinct_col}}
        #     result = self.db.get_collection(collection).find(query)
        # else:
        #     query = {f'{key}' : {'$in': distinct_col}}
        #     result = self.db.get_collection(collection).find(query, projection)
        return result

    def close(self):
        if self.con is not None:
            self.con.close()
    def __del__(self):
        if self.con is not None:
            self.close()






# conn = pymongo.MongoClient("mongodb://{}:{}@{}:{}/{}".format('chatbot', 'funnywork1018!', '54.180.9.189', int('27017'), 'healthcare_chatbot'))
# db = conn.get_database('healthcare_chatbot')
# db.list_collection_names()
# print(db.list_collection_names())
# col = db['hidoc']
# mydict = [{ "mb_name": "homzzang", "mb_level": "10" }]
# x = col.insert_many(mydict)
# print(x)