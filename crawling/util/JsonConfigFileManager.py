# https://blog.naver.com/PostView.nhn?isHttpsRedirect=true&blogId=wideeyed&logNo=221540272941

from easydict import EasyDict
import json
import os

class JsonConfigFileManager:
    def __init__(self, file_path):
        self.file_path = file_path
        self.values = EasyDict()
        # 경로에 파일이 없으면
        if not os.path.isfile(self.file_path):
            # self.update()
            self.export(self.create(self.file_path))
            self.reload()
        # 경로에 파일이 있으면
        else:
            self.reload()

    def reload(self):
        self.clear()
        if self.file_path:
            with open(self.file_path, 'r') as f:
                self.values.update(json.load(f))

    def clear(self):
        self.values.clear()

    def create(self, file_path):
        with open(file_path, 'w') as f:
            json.dump(dict(self.values), f, indent='\t')

    def update(self, in_dict):
        for (k1, v1) in in_dict.items():
            if isinstance(v1, dict):
                for (k2, v2) in v1.items():
                    if isinstance(v2, dict):
                        for (k3, v3) in v2.items():
                            self.values[k1][k2][k3] = v3
                    else:
                        self.values[k1][k2] = v2
            else:
                self.values[k1] = v1

    def export(self, save_file_name):
        if save_file_name:
            with open(save_file_name, 'w') as f:
                json.dump(dict(self.values), f, indent='\t')