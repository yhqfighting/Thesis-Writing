# -*- coding: utf-8 -*-
import os
import pprint
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import jieba
from scipy.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud


# 数据获取（从文件中读取）
def readFile(file_path):
    content = []
    with open(file_path, encoding="utf-8") as f:
        content = f.read()
    return content

# 数据清理（分词和去掉停用词）
def cleanWord(content):
    # 分词
    seg = jieba.jieba()
    text = seg.cut(content)

    # 读取停用词
    stopwords = []
    with open("stopwords/哈工大停用词表.txt", encoding="utf-8") as f:
        stopwords = f.read()

    new_text = []
    # 去掉停用词
    for w in text:
        if w not in stopwords:
            new_text.append(w)

    return new_text

# 数据整理（统计词频）
def statisticalData(text):
    # 统计每个词的词频
    counter = Counter(text)
    # 输出词频最高的15个单词
    pprint.pprint(counter.most_common(15))

# 数据可视化（生成词云）
def drawWordCloud(text, file_name):
    wl_space_split = " ".join(text)

    # 设置词云背景图
    b_mask = plt.imread('assets/img/bg.jpg')
    # 设置词云字体（若不设置则无法显示中文）
    font_path = 'assets/font/FZZhuoYTJ.ttf'
    # 进行词云的基本设置（背景色，字体路径，背景图片，词间距）
    wc = WordCloud(background_color="white",
                   font_path=font_path, mask=b_mask, margin=5)
    # 生成词云
    wc.generate(wl_space_split)
    # 显示词云
    plt.imshow(wc)
    plt.axis("off")
    plt.show()
    # 将词云图保存到本地
    path = os.getcwd()+'/output/'
    wc.to_file(path+file_name)

# 计算文本相似度
def calculateSimilarity(s1, s2):
    def add_space(s):
            return ' '.join(cleanWord(s))
    
    # 将字中间加入空格
    s1, s2 = add_space(s1), add_space(s2)
    # 转化为TF矩阵
    cv = TfidfVectorizer(tokenizer=lambda s: s.split())
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()
    # 计算TF系数
    return np.dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1]))

if __name__ == "__main__":
    content_zhai = readFile('data/zhai.txt')
    content_chen=readFile('data/chen.txt')

    # 对翟天临文章的分析
    text_zhai = cleanWord(content_zhai)
    statisticalData(text_zhai)
    drawWordCloud(text_zhai, 'zhai.png')


    # 对陈坤文章的分析
    text_chen = cleanWord(content_chen)
    statisticalData(text_chen)
    drawWordCloud(text_chen, 'chen.png')

    print('两篇文章的相似度为：')
    print(calculateSimilarity(content_chen,content_zhai))
