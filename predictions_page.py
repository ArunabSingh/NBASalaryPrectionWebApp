import streamlit as st
import pickle
import numpy as np
import warnings
import locale

warnings.filterwarnings("ignore")

regressor = pickle.load(open('model.pkl', 'rb'))

locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')


def show_predict_page():
    st.title("NBA Players Salary Prediction")
    # h3 heading for the text below.
    st.write("""### We need some information to predict the salary""")

    ppg = st.slider("Points Per Game", 0.0, 50.0, 3.0)
    apg = st.slider("Assists Per Game", 0.0, 30.0, 3.0)
    mpg = st.slider("Minutes Played Per Game", 0, 50, 3)
    spg = st.slider("Steals Per Game", 0.0, 12.0, 1.0)
    bpg = st.slider("Blocks Per Game", 0.0, 18.0, 2.0)
    ws = st.slider("Win Shares", -1.0, 15.0, 1.0)
    age = st.slider("Age", 18, 45, 21)

    ok = st.button("Calculate Salary")

    if ok:
        X = np.array([[ppg, apg, mpg, spg, bpg, ws, age]])
        # X = X.astype(float)

        salary = regressor.predict(X)
        salary = (np.round(np.exp(salary[0]), 0)) * 10
        salary = locale.currency(salary[0], grouping=True)
        st.subheader('The Salary prediction for the NBA player is: {}'.format(salary))
