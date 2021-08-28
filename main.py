# coding:utf-8
import json
from base_fasttext.fasttext_classify import train_three_class
from base_fasttext.fasttext_classify import train_all_class


def main(train_class = 'three'):
    if train_class == 'three':
        # 使用三个类别的数据训练
        three_classifier = train_three_class()
        three_classifier_map_path = './base_fasttext/data/three_class/map.json'
        with open(three_classifier_map_path, 'r', encoding='utf-8') as f:
            three_classifier_map = json.load(f)
        true_class = '工业技术'
        test_data = '通信 原理 - ( 第 3 版 )   本书 系统地 介绍 通信 的 基本概念 、 基本 理论 和 基本 分析方法 。 在 保持 一定 理论 深度 的 基础 上 ， 本书 尽可能 简化 数学分析 过程 ， 突出 对 概念 、 新 技术 的 介绍 ； 叙述 上 力求 概念 清楚 、 重点 突出 、 深入浅出 、 通俗易懂 ； 内容 上 力求 科学性 、 先进性 、 系统性 与 实用性 的 统一 。   本书 共 10 章 ， 内容 包括 ： 绪论 、 信号 与 噪声 分析 、 模拟 调制 系统 、 模拟信号 的 数字传输 、 数字信号 的 基带 传输 、 数字信号 的 载波 传输 、 现代 数字 调制 技术 、 信道 、 信道编码 和 扩频通信 。 内容 涵盖 国内 通信 原理 教学 的'.encode('utf-8')
        result = three_classifier.predict(str(test_data))[0][0]
        predicT_class = list(three_classifier_map.keys())[list(three_classifier_map.values()).index(int(result[-1]))]
        print('预测的类别为：{}'.format(predicT_class))
        print('真实的类别为：{}'.format(true_class))
    else:
        # 使用所有数据训练
        all_classifier = train_all_class()
        all_classifier_map_path = './base_fasttext/data/all_class/map.json'
        with open(all_classifier_map_path, 'r', encoding='utf-8') as f:
            all_classifier_map = json.load(f)
        true_class = '工业技术'
        test_data = '通信 原理 - ( 第 3 版 )   本书 系统地 介绍 通信 的 基本概念 、 基本 理论 和 基本 分析方法 。 在 保持 一定 理论 深度 的 基础 上 ， 本书 尽可能 简化 数学分析 过程 ， 突出 对 概念 、 新 技术 的 介绍 ； 叙述 上 力求 概念 清楚 、 重点 突出 、 深入浅出 、 通俗易懂 ； 内容 上 力求 科学性 、 先进性 、 系统性 与 实用性 的 统一 。   本书 共 10 章 ， 内容 包括 ： 绪论 、 信号 与 噪声 分析 、 模拟 调制 系统 、 模拟信号 的 数字传输 、 数字信号 的 基带 传输 、 数字信号 的 载波 传输 、 现代 数字 调制 技术 、 信道 、 信道编码 和 扩频通信 。 内容 涵盖 国内 通信 原理 教学 的'.encode('utf-8')
        result = all_classifier.predict(str(test_data))[0][0]
        predicT_class = list(all_classifier_map.keys())[list(all_classifier_map.values()).index(int(result[-1]))]
        print('预测的类别为：{}'.format(predicT_class))
        print('真实的类别为：{}'.format(true_class))

main(train_class='three')

