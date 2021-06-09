
import pandas as pd
import numpy as np
import pickle
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)
plt.axis("off")
#plt.figure( figsize = (15, 10))
plt.tight_layout(pad = 0)

def random_color_func(word=None, font_size=None, position=None,  orientation=None, font_path=None, random_state=None):
    h = int(360.0 * 21.0 / 255.0)
    s = int(100.0 * 255.0 / 255.0)
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
#header
st.write("Versione beta 1.0")
#sidebar
st.sidebar.image('data/logo-regione.jpg')

platform_switch = st.sidebar.selectbox(
    'Seleziona la piattaforma',
    piattaforme
)

polarity_switch = st.sidebar.radio(
    'Seleziona per mostrare aspetti positivi e negativi',
    ('Positivo', 'Negativo')
)

if platform_switch:
    #load platdorm
    temp = data[data['piattaforma'] == platform_switch]
    try:
        pipeline.fit(temp['commento_p'], temp['target'])
    except:
        st.write("too small size")

    #generate word lists
    index_positive = np.argsort(-1*pipeline[1].coef_)[0][: n_words]
    positive_words = ""
    for i in index_positive:
        positive_words += " " + pipeline[0].get_feature_names()[i].replace(" ", '_')
     
    index_negative = np.argsort(pipeline[1].coef_)[0][: n_words]
    negative_words = ""
    for i in index_negative:
        negative_words += " " + pipeline[0].get_feature_names()[i].replace(" ", '_')

    #generate wordcloud
    if polarity_switch == "Positivo":
        wordcloud = WordCloud(background_color='white').generate(positive_words)     
    else:
        wordcloud = WordCloud(background_color='white',
        color_func=random_color_func).generate(negative_words)
    plt.imshow(wordcloud, interpolation = 'bilinear')
    st.pyplot()


