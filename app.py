import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import jieba
from collections import Counter
from pyecharts.charts import WordCloud
from pyecharts import options as opts

# 加载自定义停用词表
def load_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        stopwords = set(line.strip() for line in f)
    return stopwords

# 分词并去除停用词
def segment(text, stopwords):
    # 使用 jieba 进行分词，使用精确模式
    words = list(jieba.cut(text, cut_all=False))  # 使用精确模式

    # 去除停用词和空白字符
    filtered_words = [word for word in words if word not in stopwords and word.strip()]

    return filtered_words

# 设置页面标题
st.title("文本分析工具")

# 用户输入文章URL
url = st.text_input('文章URL')

if url:
    # 请求URL获取文本内容
    response = requests.get(url)
    response.encoding = "utf-8"
    soup = BeautifulSoup(response.text, 'html.parser')

    # 提取标题和内容
    title = soup.title.string if soup.title else "无标题"
    content = soup.get_text()

    # 清洗文本：移除HTML标签和特殊符号
    clean_text = re.sub(r'[^\w\s]', '', re.sub('<[^>]+>', '', content))

    # 加载停用词
    stopwords_file = r'E:\Python实训\stopwords.txt'  # 停用词文件路径
    stopwords = load_stopwords(stopwords_file)

    # 分词并过滤
    word_list = segment(clean_text, stopwords)

    # 统计词频
    counts = Counter(word_list)
    items = list(counts.items())
    items.sort(key=lambda x: x[1], reverse=True)

    # 创建词频列表（每个元素为 (word, frequency) 的元组）
    word_freq_all = [(word, freq) for word, freq in items]  # 所有词频
    top_words_freq = word_freq_all[:20]  # 只保留前20个高频词

    # 默认展示所有词汇词频
    show_top_only = False

    if st.button('过滤低频词'):
        show_top_only = True
    if st.button('取消过滤低频词') and show_top_only == True:
        show_top_only = False

    # 根据按钮状态选择要展示的词频数据
    word_freq_to_show = top_words_freq if show_top_only else word_freq_all

    # 创建词云图
    def create_word_cloud(shape):
        word_cloud = (
            WordCloud()
            .add("", word_freq_to_show, word_size_range=[20, 100], shape=shape)
            .set_global_opts(
                title_opts=opts.TitleOpts(title="词云图"),
                tooltip_opts=opts.TooltipOpts(is_show=True),
            )
        )
        return word_cloud



    # 侧边栏中选择词云图形状
    shapes = ["circle", "square", "diamond", "triangle-up", "triangle-down", "pin", "star"]
    selected_shape = st.sidebar.radio("词云图形状:", shapes)

    # 创建词云图
    word_cloud = create_word_cloud(selected_shape)

    # 在 Streamlit 中显示词云图
    st.components.v1.html(word_cloud.render_embed(), height=600)

    # 输出前20个出现最多的单词及其次数
    if show_top_only:
        st.write("前20个高频词:")
    else:
        st.write("所有词汇词频:")

    for i, (word, count) in enumerate(word_freq_to_show):
        st.write(f"{i + 1}. {word}: {count}次")

    # 侧边栏中选择数据分类
    st.divider()
