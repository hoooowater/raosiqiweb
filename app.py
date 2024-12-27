import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import jieba
from collections import Counter
from pyecharts.charts import WordCloud, Line, Bar, Pie
from pyecharts import options as opts
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 加载自定义停用词表
def load_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        stopwords = set(line.strip() for line in f)
    return stopwords

# 分词并去除停用词
def segment(text, stopwords):
    words = list(jieba.cut(text, cut_all=False))
    filtered_words = [word for word in words if word not in stopwords and word.strip()]
    return filtered_words

# 创建不同类型的图表
def create_line_chart(word_freq):
    line = (
        Line()
        .add_xaxis([word for word, freq in word_freq])
        .add_yaxis("频率", [freq for word, freq in word_freq])
        .set_global_opts(title_opts=opts.TitleOpts(title="词汇频率折线图"))
    )
    return line

def create_bar_chart(word_freq):
    bar = (
        Bar()
        .add_xaxis([word for word, freq in word_freq])
        .add_yaxis("频率", [freq for word, freq in word_freq])
        .set_global_opts(title_opts=opts.TitleOpts(title="词汇频率柱状图"))
    )
    return bar

def create_pie_chart(word_freq):
    pie = (
        Pie()
        .add("", word_freq)
        .set_global_opts(title_opts=opts.TitleOpts(title="词汇频率饼图"))
        .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c} ({d}%)"))
    )
    return pie



def create_word_cloud(shape, word_freq):
    word_cloud = (
        WordCloud()
        .add("", word_freq, word_size_range=[20, 100], shape=shape)
        .set_global_opts(
            title_opts=opts.TitleOpts(title="词云图"),
            tooltip_opts=opts.TooltipOpts(is_show=True),
        )
    )
    return word_cloud

# 创建 Seaborn 统计图表
def create_seaborn_regplot(df):
    plt.figure(figsize=(10, 6))
    sns.regplot(x="length", y="frequency", data=df)

    st.pyplot(plt)

def create_seaborn_distplot(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df["frequency"], kde=True)

    st.pyplot(plt)

def create_seaborn_pairplot(df):
    plt.figure(figsize=(10, 8))
    sns.pairplot(df, vars=["length", "frequency"])

    st.pyplot(plt)

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
    stopwords_file = r'stopwords.txt'  # 停用词文件路径
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

    # 将词频转换为DataFrame
    df_word_freq = pd.DataFrame(word_freq_all, columns=['word', 'frequency'])
    df_word_freq['length'] = df_word_freq['word'].apply(len)

    # 默认展示所有词汇词频
    show_top_only = False

    if st.button('过滤低频词'):
        show_top_only = True
    if st.button('取消过滤低频词') and show_top_only == True:
        show_top_only = False

    # 根据按钮状态选择要展示的词频数据
    word_freq_to_show = top_words_freq if show_top_only else word_freq_all
    # 根据按钮状态选择要展示的词频数据，并重置索引以从1开始
    if show_top_only:
        df_word_freq_to_show = df_word_freq.head(20).reset_index(drop=True)
    else:
        df_word_freq_to_show = df_word_freq.reset_index(drop=True)

    # 设置索引从1开始
    df_word_freq_to_show.index = df_word_freq_to_show.index + 1

    # 侧边栏中选择图表类型
    chart_types = ["词云图", "折线图", "柱状图", "饼图",  "回归图", "直方图", "成对关系图"]
    selected_chart_type = st.sidebar.selectbox("选择图表类型:", chart_types)

    # 显示表格选项
    show_table = st.sidebar.checkbox("显示词频表格")

    # 显示动态线图选项
    show_dynamic_line_chart = st.sidebar.checkbox("显示动态线图")

    # 根据选择创建对应的图表
    if selected_chart_type == "词云图":
        shapes = ["circle", "square", "diamond", "triangle-up", "triangle-down", "pin", "star"]
        selected_shape = st.sidebar.radio("词云图形状:", shapes)
        chart = create_word_cloud(selected_shape, word_freq_to_show)
    elif selected_chart_type == "折线图":
        chart = create_line_chart(word_freq_to_show)
    elif selected_chart_type == "柱状图":
        chart = create_bar_chart(word_freq_to_show)
    elif selected_chart_type == "饼图":
        chart = create_pie_chart(word_freq_to_show)
    elif selected_chart_type == "回归图":
        create_seaborn_regplot(df_word_freq_to_show)
    elif selected_chart_type == "直方图":
        create_seaborn_distplot(df_word_freq_to_show)
    elif selected_chart_type == "成对关系图":
        create_seaborn_pairplot(df_word_freq_to_show)

    # 如果不是Seaborn图表，则渲染pyecharts图表
    if selected_chart_type not in ["回归图", "直方图", "成对关系图"]:
        st.components.v1.html(chart.render_embed(), height=600)

    # 输出前20个出现最多的单词及其次数
    if show_top_only:
        st.write("前20个高频词:")
    else:
        st.write("所有词汇词频:")

    for i, (word, count) in enumerate(word_freq_to_show):
        st.write(f"{i + 1}. {word}: {count}次")

    # 添加空行
    st.write("")

    # 显示表格
    if show_table:
        st.dataframe(df_word_freq_to_show)

    # 显示动态线图
    if show_dynamic_line_chart:
        st.line_chart(df_word_freq_to_show.set_index('word')['frequency'])
