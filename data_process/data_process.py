import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter

def extract_three_cls_data(data_path,save_path, txt_save_path):
    map_path = './base_fasttext/data/three_class/map.json'
    data = pd.read_csv(data_path, sep='\t')
    cls_data = data[(data['label'] == '童书') | (data['label'] == '工业技术') | (data['label'] == '大中专教材教辅')]
    cls_data.index = range(len(cls_data))
    print(Counter(cls_data['label']))
    print('总共 {} 个类别'.format(len(np.unique(cls_data['label']))))
    label_map = {key:index for index, key in enumerate(np.unique(cls_data['label']))}
    label_map_json = json.dumps(label_map, ensure_ascii=False, indent=3)
    if not os.path.exists(label_map_json):
        with open(map_path, 'w', encoding='utf-8') as f:
            f.write(label_map_json)
    cls_data['fasttext_label'] = cls_data['label'].map(label_map)
    for i in range(len(cls_data['fasttext_label'])):
        cls_data['fasttext_label'][i] = '__label__{}'.format(cls_data['fasttext_label'][i])
    print(len(cls_data))
    with open('./data/stopwords.txt', 'r', encoding='utf-8') as f:
        stopwords = f.readlines()
        stopwords = [i.strip() for i in stopwords]
    cls_data.to_csv(save_path, index=False)
    with open(txt_save_path, 'a+', encoding='utf-8') as f:
        for idx,row in tqdm(cls_data.iterrows(), desc='去除停用词：', total=len(cls_data)):
            words = row['text'].split(' ')
            out_str = ''
            for word in words:
                if word not in stopwords:
                    out_str += word
                    out_str += ' '
            row['text'] = out_str.encode('utf-8')

            line = str(row['text']) + '\t' + row['fasttext_label'] + '\n'
            f.write(line)

def extract_all_cls_data(data_path,save_path, txt_save_path):
    map_path = './base_fasttext/data/all_class/map.json'
    data = pd.read_csv(data_path, sep='\t')
    cls_data = data
    cls_data.index = range(len(cls_data))
    print(Counter(cls_data['label']))
    print('总共 {} 个类别'.format(len(np.unique(cls_data['label']))))
    label_map = {key:index for index, key in enumerate(np.unique(cls_data['label']))}
    label_map_json = json.dumps(label_map, ensure_ascii=False, indent=3)
    if not os.path.exists(label_map_json):
        with open(map_path, 'w', encoding='utf-8') as f:
            f.write(label_map_json)
    cls_data['fasttext_label'] = cls_data['label'].map(label_map)
    for i in range(len(cls_data['fasttext_label'])):
        cls_data['fasttext_label'][i] = '__label__{}'.format(cls_data['fasttext_label'][i])
    print(len(cls_data))
    with open('./data/stopwords.txt', 'r', encoding='utf-8') as f:
        stopwords = f.readlines()
        stopwords = [i.strip() for i in stopwords]
    cls_data.to_csv(save_path, index=False)
    with open(txt_save_path, 'a+', encoding='utf-8') as f:
        for idx,row in tqdm(cls_data.iterrows(), desc='去除停用词：', total=len(cls_data)):
            words = row['text'].split(' ')
            out_str = ''
            for word in words:
                if word not in stopwords:
                    out_str += word
                    out_str += ' '
            row['text'] = out_str.encode('utf-8')
            line = str(row['text']) + '\t' + row['fasttext_label'] + '\n'
            f.write(line)
