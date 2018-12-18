import json
import os
import re

from pythainlp import word_tokenize
from pythainlp.sentiment import sentiment


def process_word_to_feature(text):
    """
    Process thai text to sparse feature dict
    :param text: text for process
    :return: dict after process
    """
    word_list = word_tokenize(text.strip(), engine='newmm')
    while ' ' in word_list:
        word_list.remove(' ')
    sentiment_str = sentiment(text.strip())
    if sentiment_str == 'neg':
        senti = 1
    elif sentiment_str == 'neutral':
        senti = 2
    elif sentiment_str == 'pos':
        senti = 3
    # text_dict = {'__len': len(temp), '__word': len(word_list), '__sentiment': senti}
    text_dict = {'__sentiment': senti}
    for temp in word_list:
        if temp in text_dict:
            text_dict[str(temp)] += 1
        else:
            text_dict[str(temp)] = 1
    return text_dict


class CleanData:

    def __init__(self):
        self.file = []
        self.feature_list = []

    def add_line_file(self, file_name):
        """
        Add line file for process data

        :param file_name: file name of line message file (Download from line PC)
        :return:
        """
        self.file.append((open(file_name, 'r', encoding="utf8"), 'line'))

    def add_fb_main_dir(self, main_path):
        """
        Add all facebook file in this path for process data

        :param main_path:  main path of facebook message dilatory (Download from process fie facebook system in setting page)
        :return:
        """
        all_files = os.listdir(main_path)
        for folder in all_files:
            self.file.append((open(main_path + '/' + folder + '/message.json', 'r', encoding="ISO-8859-1"), 'fb'))

    def make_all_feature(self, output_file_name):
        """
        Start process all data has been added.

        :param output_file_name: output feature file name
        :return:
        """
        f = open(output_file_name, "w", encoding="utf8")
        for file in self.file:
            if len(self.feature_list) > 100000:
                break
            print(file[0])
            self.feature_list += self.make_feature(file[0], file[1])
        f.write(json.dumps(self.feature_list, ensure_ascii=False))
        f.close()

    def make_feature(self, file, type):
        """
        Make sparse feature dict for one file

        :param file: file have been open
        :param type: type of data file (Line or Facebook)
        :return: List of sparse feature
        """
        dict_list = []
        if type == 'line':
            for line in file.readlines():
                if 'สติกเกอร์' in line or 'รูป' in line or 'ยกเลิกข้อความ' in line or line == '':
                    continue
                temp = re.sub('\d\d:\d\d\s[\S]*\s', r'', line.strip())
                temp = re.sub('\d\d\d\d.\d\d.\d\d\s[\S]*', r'', temp)
                if len(temp) == 0:
                    continue
                dict_list.append(process_word_to_feature(temp))
        elif type == 'fb':
            text = file.read()
            js = json.loads(text)
            for message in js['messages'][-1000:]:
                if 'content' in message:
                    tmp = str(message['content'])
                    real_text = tmp.encode('latin-1').decode('utf-8')
                    if 'ส่งสติกเกอร์' in real_text or 'ได้ส่งรูปภาพ' in real_text or 'ได้ส่งตำแหน่งที่ตั้ง' in real_text or real_text == '':
                        continue
                    if len(real_text) == 0:
                        continue
                    dict_list.append(process_word_to_feature(real_text))
        file.close()
        return dict_list
