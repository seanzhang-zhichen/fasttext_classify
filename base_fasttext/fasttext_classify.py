from genericpath import exists
import os
import fasttext
from data_process.data_process import extract_three_cls_data
from data_process.data_process import extract_all_cls_data

def train_three_class():
    train_data_path = './data/train.csv'
    train_csv_path = './base_fasttext/data/three_class/train.csv'
    train_txt_path = './base_fasttext/data/three_class/train.txt'
    if not os.path.exists(train_txt_path):
        extract_three_cls_data(train_data_path, train_csv_path, train_txt_path)
    test_data_path = './data/test.csv'
    test_csv_path = './base_fasttext/data/three_class/test.csv'
    test_txt_path = './base_fasttext/data/three_class/test.txt'
    if not os.path.exists(test_txt_path):
        extract_three_cls_data(test_data_path, test_csv_path, test_txt_path)
    dev_data_path = './data/dev.csv'
    dev_csv_path = './base_fasttext/data/three_class/dev.csv'
    dev_txt_path = './base_fasttext/data/three_class/dev.txt'
    if not os.path.exists(dev_txt_path):
        extract_three_cls_data(dev_data_path, dev_csv_path, dev_txt_path)
    # classifier = fasttext.train_supervised(input= train_txt_path, autotuneValidationFile = dev_txt_path)
    model_path = './base_fasttext/model/fasttext_three_class.pkl'
    if not os.path.exists(model_path):
        classifier = fasttext.train_supervised(train_txt_path,
                                                label="__label__",
                                                dim=100,
                                                epoch=10,
                                                lr=0.1,
                                                wordNgrams=3,
                                                loss='softmax',
                                                thread=8,
                                                verbose=True,
                                                minCount = 5)
        classifier.save_model(model_path)
        result = classifier.test(test_txt_path)
        print('F1 Score: {}'.format(result[1] * result[2] * 2 / (result[2] + result[1])))
    else:
        classifier = fasttext.load_model(model_path)
        # result = classifier.test(test_txt_path)
        # print('F1 Score: {}'.format(result[1] * result[2] * 2 / (result[2] + result[1])))
    return classifier



def train_all_class():
    train_data_path = './data/train.csv'
    train_csv_path = './base_fasttext/data/all_class/train.csv'
    train_txt_path = './base_fasttext/data/all_class/train.txt'
    if not os.path.exists(train_txt_path):
        extract_all_cls_data(train_data_path, train_csv_path, train_txt_path)
    test_data_path = './data/test.csv'
    test_csv_path = './base_fasttext/data/all_class/test.csv'
    test_txt_path = './base_fasttext/data/all_class/test.txt'
    if not os.path.exists(test_txt_path):
        extract_all_cls_data(test_data_path, test_csv_path, test_txt_path)
    dev_data_path = './data/dev.csv'
    dev_csv_path = './base_fasttext/data/all_class/dev.csv'
    dev_txt_path = './base_fasttext/data/all_class/dev.txt'
    if not os.path.exists(dev_txt_path):
        extract_all_cls_data(dev_data_path, dev_csv_path, dev_txt_path)
    # classifier = fasttext.train_supervised(input= train_txt_path, autotuneValidationFile = dev_txt_path)
    model_path = './base_fasttext/model/fasttext_all_class.pkl'
    if not os.path.exists(model_path):
        classifier = fasttext.train_supervised(train_txt_path,
                                                label="__label__",
                                                dim=100,
                                                epoch=10,
                                                lr=0.1,
                                                wordNgrams=3,
                                                loss='softmax',
                                                thread=8,
                                                verbose=True,
                                                minCount = 5)
        classifier.save_model(model_path)
        result = classifier.test(test_txt_path)
        print('F1 Score: {}'.format(result[1] * result[2] * 2 / (result[2] + result[1])))
    else:
        classifier = fasttext.load_model(model_path)
        # result = classifier.test(test_txt_path)
        # print('F1 Score: {}'.format(result[1] * result[2] * 2 / (result[2] + result[1])))
    return classifier