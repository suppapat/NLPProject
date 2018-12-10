import json
import re

from pythainlp import word_tokenize
from pythainlp.sentiment import sentiment

class cleandata():
#file = open('NLP/NLPProject/NLPProject/[LINE]หารแหลมเกต.txt', 'r',encoding="utf8")

    def __init__(self):
        self.file=[]
        self.feature_list=[]

    def add_file(self,file_name):
        self.file.append(open(file_name, 'r', encoding="utf8"))

    def make_all_feature(self,output_file_name):
        for file in self.file:
            self.feature_list += self.make_feature(file)
        f = open(output_file_name, "x")
        f.write(json.dumps(self.feature_list, ensure_ascii=False,encoding="utf8"))
        f.close()

    def close_all_files(self):
        for file in self.file:
            file.close()

    def make_feature(self,file):
        dict_list = []
        for line in file.readlines():
            if 'สติกเกอร์' in line or 'รูป' in line or 'ยกเลิกข้อความ' in line or line=='':
                continue
            temp = re.sub('\d\d:\d\d\s[\S]*\s', r'', line.strip())
            temp = re.sub('\d\d\d\d.\d\d.\d\d\s[\S]*', r'', temp)
            if len(temp)==0:
                continue
            word_list = word_tokenize(temp.strip(), engine='newmm')
            while ' ' in word_list:
                word_list.remove(' ')
            sentiment_str= sentiment(temp.strip())
            if sentiment_str=='neg':
                senti=1
            elif sentiment_str=='neutral':
                senti=2
            elif sentiment_str=='pos':
                senti=3
            text_dict = {'__len': len(temp), '__word': len(word_list), '__sentiment': senti}
            for temp in word_list:
                if temp in text_dict:
                    text_dict[str(temp)] += 1
                else:
                    text_dict[str(temp)] = 1
            dict_list.append(text_dict)
        return dict_list
