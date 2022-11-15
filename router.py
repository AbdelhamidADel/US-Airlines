import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import base64
from sklearn import  svm
from clean import clean_the_tweet ,text_process
import pickle

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="centered",page_icon="✈️",page_title="US Airlines")
# --------------------------------------------------------------------
hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
#----------------------------------------------------------------
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
set_background('plane background.jpg')
# --------------------------------------------------Prediction PAGE-----------------------------------------------------------
DATA_URL = (
    "Tweets.csv"
)

st.title("Sentiment Analysis of Tweets about US Airlines")
st.sidebar.markdown("<h1 style='text-align: center; color: white;'>Powered By Data Scientist Abdelhamid Adel</h1>", unsafe_allow_html=True)
st.sidebar.caption("<h5 style='text-align: center;'><a href='https://github.com/AbdelhamidADel'>My GitHub</a></h5>", unsafe_allow_html=True)
styl = """
    <style>
      a {
  box-shadow: inset 0 0 0 0 #54b3d6;
  color: #54b3d6;
  margin: 0 -.25rem;
  padding: 0 .25rem;
  transition: color .3s ease-in-out, box-shadow .3s ease-in-out;
}
a:hover {
  box-shadow: inset 200px 0 0 0 #54b3d6;
  color: white;
}
    </style>        
    """
st.markdown(styl, unsafe_allow_html=True)

@st.cache(persist=True)
def load_data():
    data = pd.read_csv(DATA_URL)
    data['tweet_created'] = pd.to_datetime(data['tweet_created'])
    return data

data = load_data()
# ==============================================================================================================
st.sidebar.subheader("Show random tweet")
random_tweet = st.sidebar.radio('Sentiment', ('positive', 'negative'))
st.sidebar.markdown(data.query("airline_sentiment == @random_tweet")[["text"]].sample(n=1).iat[0, 0])
# ==============================================================================================================
# ============================================  Prediction  =================================================
# ==============================================================================================================

st.sidebar.subheader("Tweet Sentiment Prediction")
sv_opt=svm.SVC(probability=True, C=10, kernel='rbf', gamma='scale')
check=st.sidebar.text_area(label="input tweet here....")
if check=="":
    st.sidebar.warning('please fill the field before click predict')
else  :
    clean_cheeck=clean_the_tweet(check)
    clean_cheeck2=text_process(clean_cheeck)
    vectorizer = pickle.load(open("vectorizer.pickle", 'rb'))
    fit_model = pickle.load(open("fit_model.pickle", 'rb'))
    new=vectorizer.transform([clean_cheeck2])
    result=fit_model.predict(new)
    if st.sidebar.button('Predict'):
            if result[0]==0:
                st.sidebar.error('Negative')
            else:
                st.sidebar.success("Positive") 
    

# ==============================================================================================================
# ==============================================================================================================
# ==============================================================================================================

st.sidebar.markdown("### Number of tweets by sentiment")
select = st.sidebar.selectbox('Visualization type', ['Bar plot', 'Pie chart'], key='7')
sentiment_count = data['airline_sentiment'].value_counts()
sentiment_count = pd.DataFrame({'Sentiment':sentiment_count.index, 'Tweets':sentiment_count.values})
if not st.sidebar.checkbox("Hide", True):
    st.markdown("### Number of tweets by sentiment")
    if select == 'Bar plot':
        fig = px.bar(sentiment_count, x='Sentiment', y='Tweets',color='Sentiment',template="seaborn" ,height=500)
        st.plotly_chart(fig)
    else:
        fig = px.pie(sentiment_count, values='Tweets', names='Sentiment')
        st.plotly_chart(fig)

st.sidebar.subheader("When and where are users tweeting from?")
hour = st.sidebar.slider("Hour to look at", 0, 23)
modified_data = data[data['tweet_created'].dt.hour == hour]
if not st.sidebar.checkbox("Close", True, key='1'):
    st.markdown("### Tweet locations based on time of day")
    st.markdown("%i tweets between %i:00 and %i:00" % (len(modified_data), hour, (hour + 1) % 24))
    st.map(modified_data)
    if st.sidebar.checkbox("Show raw data", False):
        st.write(modified_data)


st.sidebar.subheader("Total number of tweets for each airline")
each_airline = st.sidebar.selectbox('Visualization type', ['Bar plot', 'Pie chart'], key='2')
airline_sentiment_count = data.groupby('airline')['airline_sentiment'].count().sort_values(ascending=False)
airline_sentiment_count = pd.DataFrame({'Airline':airline_sentiment_count.index, 'Tweets':airline_sentiment_count.values.flatten()})
if not st.sidebar.checkbox("Close", True, key='3'):
    if each_airline == 'Bar plot':
        st.subheader("Total number of tweets for each airline")
        fig_1 = px.bar(airline_sentiment_count, x='Airline', y='Tweets', color='Tweets',template="plotly",height=500)
        st.plotly_chart(fig_1)
    if each_airline == 'Pie chart':
        st.subheader("Total number of tweets for each airline")
        fig_2 = px.pie(airline_sentiment_count, values='Tweets', names='Airline')
        st.plotly_chart(fig_2)


@st.cache(persist=True)
def plot_sentiment(airline):
    df = data[data['airline']==airline]
    count = df['airline_sentiment'].value_counts()
    count = pd.DataFrame({'Sentiment':count.index, 'Tweets':count.values.flatten()})
    return count



st.sidebar.subheader("Breakdown airline by sentiment")
choice = st.sidebar.multiselect('Pick airlines', ('US Airways','United','American','Southwest','Delta','Virgin America'), key=0)
if len(choice) > 0:
    choice_data = data[data.airline.isin(choice)]
    fig_0 = px.histogram(
                        choice_data, x='airline', y='airline_sentiment',
                         histfunc='count', color='airline_sentiment',
                         facet_col='airline_sentiment', labels={'airline_sentiment':'tweets'},
                          height=600, width=800)
    st.plotly_chart(fig_0)

st.sidebar.header("Word Cloud")
word_sentiment = st.sidebar.radio('Display word cloud for what sentiment?', ('positive', 'neutral', 'negative'))
if not st.sidebar.checkbox("Close", True, key='5'):
    st.subheader('Word cloud for %s sentiment' % (word_sentiment))
    df = data[data['airline_sentiment']==word_sentiment]
    words = ' '.join(df['text'])
    processed_words = ' '.join([word for word in words.split() if 'http' not in word and not word.startswith('@') and word != 'RT'])
    wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', width=800, height=640).generate(processed_words)
    plt.imshow(wordcloud)
    plt.xticks([])
    plt.yticks([])
    st.pyplot()