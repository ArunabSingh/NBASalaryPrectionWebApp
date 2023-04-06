import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import numpy as np


@st.cache_data
def load_data():
    nba = pd.read_csv("datasets/NBA_Stats_2017.csv")

    nba['PPG'] = nba['PTS'] / nba['G']
    nba['MPG'] = nba['MP'] / nba['G']
    nba['APG'] = nba['AST'] / nba['G']
    nba['SPG'] = nba['STL'] / nba['G']
    nba['BPG'] = nba['BLK'] / nba['G']

    df = nba[['PPG', 'MPG', 'APG', 'SPG', 'BPG', 'WS', 'Age', 'Salary']]
    return nba, df


original_df, df = load_data()


def show_explore_page():
    # Set page background color
    st.markdown(
        """
        <style>
        body {
            background-color: #f0f0f0;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Set title font and size
    st.title(
        "Explore NBA Player Salaries"
    )

    st.header("NBA Season 2017-2018")
    st.write("Model Goal:   We will be build a Regression model that is able to predict the yearly salaries of NBA "
             "players based on thier season statistics from the 2017-18 season.")

    st.write("Original dataset preview.")
    st.dataframe(original_df.head(10))
    st.write("After doing data manipulation and cleaning.")
    st.write("The final dataset contains 7 main features which were either engineered or taken from the original "
             "dataset.")
    st.write("These consist of the following:\n"
             "Points Per Game (PPG)\n"
             "Minutes Played Per Game (MPG)\n"
             "Assists Per Game (APG)\n"
             "Steals Per Game (SPG)\n"
             "Blocks Per Game (BPG)\n"
             "Win Shares (WS)\n"
             "Age")
    st.dataframe(df.head(10))

    # Set subtitle font and size
    st.markdown(
        """<h2 style='font-family: "Helvetica Neue", Helvetica, sans-serif; font-size: 24px; margin-top: 50px; 
        margin-bottom: 30px; text-align: center;'>
            Scatter Plots
        </h2>
         <p>
         We draw scatter plots to show the correlation between the salary and the features.
         </p>
         """,
        unsafe_allow_html=True
    )

    # Use a grid layout for the scatter plots
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(12, 8))
    axs = axs.ravel()

    for i, feature in enumerate(df.columns):
        if i < 7:
            axs[i].scatter(df[feature], df['Salary'], alpha=0.3)
            axs[i].set_xlabel(feature)
            axs[i].set_ylabel('Salary')
            axs[i].tick_params(axis='both', labelsize=12)

    # Add padding to the scatter plots
    fig.tight_layout(pad=3.0)
    st.pyplot(fig)

    # Set subtitle font and size
    st.markdown(
        """<h2 style='font-family: "Helvetica Neue", Helvetica, sans-serif; font-size: 24px; margin-top: 50px; 
        margin-bottom: 30px; text-align: center;'>
            Box Plots
        </h2>
        <p>We drew box plots for the "Age" and "Salary" column to see the outliers for these features.<br>
        The "Age" column has very less outliers and can be easily ignored, while the "Salary" columns have a bunch of outliers.
        <br>
         In the case of NBA, outliers represent legitimate data points, such as superstar players who earn significantly 
         more than the average player, then removing them could result in a less accurate model"         
         </p>
        """
        , unsafe_allow_html=True
    )

    fig, axs = plt.subplots(1, 2, figsize=(12, 8))
    axs[0].boxplot(df['Age'])
    axs[0].set_title('Box plot of age')
    axs[0].set_ylabel('Age')

    axs[1].boxplot(df['Salary'])
    axs[1].set_title('Salary')
    axs[1].set_ylabel('Salary')

    st.write(fig)

    # Set subtitle font and size
    st.markdown(
        """<h2 style='font-family: "Helvetica Neue", Helvetica, sans-serif; font-size: 24px; margin-top: 50px; 
        margin-bottom: 30px; text-align: center;'>
            Correlation Matrix
        </h2>
        <p>
        Brief observations:<br>
        Points per game, Assists per game, and Win Shares have a strong positive correlation with salary."         
         </p>
        """
        , unsafe_allow_html=True
    )
    corr = np.corrcoef(df.T)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, xticklabels=df.columns, yticklabels=df.columns, ax=ax)
    ax.set_title('Correlation matrix')
    st.pyplot(fig)

    # Set subtitle font and size
    st.markdown(
        """<h2 style='font-family: "Helvetica Neue", Helvetica, sans-serif; font-size: 24px; margin-top: 50px; 
        margin-bottom: 30px; text-align: center;'>
            Distribution of NBA Stats
        </h2>
        <p>
        Brief observations:<br>
        Salary is left skewed so we will normalise the data by taking logs for better predictions."<br>
        Some of the feature variables are also left skewed so we will be taking the logarithms for those features.           
         </p>
        """
        , unsafe_allow_html=True
    )
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))
    axs = axs.ravel()

    for i, feature in enumerate(df.columns):
        axs[i].hist(df[feature], bins=20)
        axs[i].set_xlabel(feature)
        axs[i].set_ylabel('Frequency')

    plt.tight_layout()
    st.pyplot(fig)

    st.write("After Data Cleaning, we created Regression Models with 4 different algorithms out of which Linear "
             "Regression gave the best results.")
    image = Image.open('results.png')
    st.image(image, caption="Results from the different models")

    st.write("Our final model, which we use in this ML Web App is a Linear Regression Model.")


