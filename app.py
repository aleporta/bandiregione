
import pandas as pd
import numpy as np
import pickle
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(page_title='customer satisfaction', page_icon=None, layout='wide', initial_sidebar_state='auto')

st.set_option('deprecation.showPyplotGlobalUse', False)
plt.axis("off")
plt.figure( figsize = (10, 7))
plt.tight_layout(pad = 0)


def random_color_func(word=None, font_size=None, position=None,  orientation=None, font_path=None, random_state=None):
    h = int(360.0 * 21.0 / 255.0)
    s = int(100.0 * 255.0 / 255.0)
    l = int(100.0 * float(random_state.randint(60, 120)) / 255.0)
    return "hsl({}, {}%, {}%)".format(h, s, l)

def random_color_func2(word=None, font_size=None, position=None,  orientation=None, font_path=None, random_state=None):
    h = int(360.0 * 150.0 / 255.0)
    s = int(100.0 * 10.0 / 255.0)
    l = int(100.0 * float(random_state.randint(60, 120)) / 255.0)
    return "hsl({}, {}%, {}%)".format(h, s, l)

## load models
with open('results//final_pipeline.pickle', 'rb') as file:
    pipeline = pickle.load(file)
data = pd.read_csv('results//cleaned.csv')
data.fillna(value = '', inplace=True)
n_words = 20

piattaforme = data['piattaforma'].unique()
bandi = data['bando'].unique()

#sidebar
st.sidebar.image('data//logo-regione.jpg')

platform_switch = st.sidebar.selectbox(
    'Seleziona la piattaforma',
    [elemento for elemento in piattaforme if elemento not in ['', 'sisco']]
)

polarity_switch = st.sidebar.radio(
    'Seleziona per mostrare aspetti positivi e negativi nella wordcloud',
    ('Negativo', 'Positivo', 'Neutrale')
)

#init columns layout
col1, col2 = st.beta_columns(2)

if platform_switch:
    #load platform
    temp = data[data['piattaforma'] == platform_switch]
    try:
        pipeline.fit(temp['commento_p'], temp['target'])
    except:
        st.write("too small size")


    ##pie plot    
    npositive = len(temp[temp['target'] == 2])
    nneutral = len(temp[temp['target'] == 1])
    y = np.array([npositive, nneutral, len(data) - npositive - nneutral])
    mylabels = ["Commenti positivi", "Commenti neutrali", "Commenti negativi"]
    colors = ['#32a897', 'grey',  '#eb6b34'] #blue, grey, red
    plt.pie(y, labels = mylabels, colors = colors, explode = [0.1 , 0.1, 0.1], radius=0.8)
    col1.pyplot() 

    #generate word lists
    index_positive = np.argsort(pipeline[1].coef_)[0][: n_words]
    positive_words = ""
    for i in index_positive:
        positive_words += " " + pipeline[0].get_feature_names()[i].replace(' ', '_')
    
    neutral_words = ""    
    index_neutral = np.argsort(-1*pipeline[1].coef_)[0]
    index_neutral = index_neutral[(int(len(index_neutral)/2 )- int(n_words/2)) : (int(len(index_neutral)/2) + int(n_words/2))]
    for i in index_neutral:
        neutral_words += " " + pipeline[0].get_feature_names()[i].replace(' ', '_')
     
    index_negative = np.argsort(-1*pipeline[1].coef_)[0][: n_words]
    negative_words = ""
    for i in index_negative:
        negative_words += " " + pipeline[0].get_feature_names()[i].replace(' ', '_')

    #generate wordcloud
    if polarity_switch == "Positivo":
        wordcloud = WordCloud(background_color='white', width = 400, height=400).generate(positive_words)  
    elif polarity_switch == "Neutrale":
    	wordcloud = WordCloud(background_color='white', width = 400, height=400,
        color_func=random_color_func2).generate(neutral_words)  
    elif polarity_switch == "Negativo":
        wordcloud = WordCloud(background_color='white',
        color_func=random_color_func, width = 400, height=400).generate(negative_words)
    plt.axis("off")
    plt.imshow(wordcloud, interpolation = 'bilinear')
    col2.pyplot()

st.sidebar.write('versione beta: 1.1')

