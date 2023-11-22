import wikipedia
from src.classes.qadataset import QADataset
from typing import List
from multiprocessing import Process, Manager
import joblib
import warnings

class MyProcess(Process):
    def __init__(self, batch: List[str], result_dict):
        super().__init__()
        self.batch = batch
        self.result_dict = result_dict
        
    def run(self):
        # 멀티프로세싱에서 실행할 코드 작성
        for title in self.batch:
            norm_title = title.replace("_", " ")
            norm_title = norm_title.replace("%27", "'")
            norm_title = norm_title.replace("%26", "&")
            try:
                wiki_text = wikipedia.page(norm_title, auto_suggest=False).content
                self.result_dict[title] = wiki_text
            except:
                print("ERROR:", title)   
        
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    manager = Manager()
    result_dict = manager.dict()
    process_list = []
    dataset = QADataset.custom_load("SquadDataset")
    titles = list(set([ex.title for ex in dataset.examples]))
    print("TITLES:", len(titles))
    batch_size = len(titles) // 32
    batch = []
    for i in range(0, len(titles), batch_size):
        if i == 32:
            batch_data = titles[i:]
        else:
            batch_data = titles[i:i+batch_size]
        batch.append(batch_data)

    for i in range(32):
        process_list.append(MyProcess(batch[i], result_dict))

    for process in process_list:
        process.start()

    for process in process_list:
        process.join()
    
    result_list = list(result_dict.keys())
    for title in titles:
        if title not in result_list:
            print(title, end=" ")

    joblib.dump({k:v for k,v in result_dict.items()}, "datasets/title_to_wiki")
    print()
    print("SUCCESS")