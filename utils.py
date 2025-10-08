from collections import defaultdict
import pandas as pd
import os

class Store2DResult():
    def __init__(self, name:str, store_dir_path:str) -> None:
        self.name = name
        self.data = defaultdict(lambda : {})
        self.store_dir_path = store_dir_path

    def insert(self, column, row, value):
        """inserting a value, at (row, column)"""
        self.data[column][row] = value

    def dump(self, filename=None):
        """dump to a file"""
        filename = self.name+".tsv"  if filename is None else filename
        filepath = os.path.join(self.store_dir_path, filename)
        self.df_data = pd.DataFrame(self.data)
        self.df_data.to_csv(filepath, sep="\t", index='infer', encoding='utf-8')
    
    def __del__(self):
        """"""
        print(f"{self.name}:: is closing.")