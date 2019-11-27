# Tweet Analysis -- objective, predict the retweet level of a tweet
import string
import seaborn as sns
sns.set()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# import the vaderSentiment we will use to get the sentiment score
import sklearn
from matplotlib import ticker
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()

# import warnings filter
from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


# function to get the vaderSentiment score
def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    return score
    # print("{:-<40} {}".format(sentence, str(score)))


# check if /data directory exists / If not then create "data" directory
data_dir_exists = os.path.isdir('./data')
if not data_dir_exists:
    os.mkdir('data')

# read / load  the tweet dataset file --- in real world, the idea is to get the tweet data through Twitter API
# For my project though, the dataset is already available in CSV format that I saved in "data" directory.

df = pd.read_csv(filepath_or_buffer='data/tweets.csv',
                 sep=',',
                 header=0)  # header starts in first line

# clean data --  There are 28 columns in this dataset with 6444 rows.  Most of the columns are not relevant to my
#  analysis.  Therefore, let's do some dropping of columns:

# drop columns that we will not use
df.drop(['id', 'is_retweet', 'original_author', 'in_reply_to_screen_name', 'in_reply_to_status_id'
            , 'in_reply_to_user_id', 'is_quote_status', 'lang', 'longitude', 'latitude', 'place_id', 'place_full_name'
            , 'place_name', 'place_type', 'place_country_code', 'place_country', 'place_contained_within'
            , 'place_attributes', 'place_bounding_box', 'source_url', 'truncated', 'entities', 'extended_entities']
        , axis=1, inplace=True)

# Now, we have a total of five (5) columns remaining as follows:
#   handle -- twitter account (HillaryClinton and realDonaldTrump)
#   text   -- the actual tweet
#   time   -- the date and time of posting
#   retweet_count -- total times the tweet was retweeted
#   favorite_count -- total times the tweet was tagged as favorite

#   Create new columns (feature extract)

# actual date column
df['actual_date'] = df['time'].str[:10]
df['actual_date'] = pd.to_datetime(df['actual_date'], format='%Y/%m/%d')


# actual time
df['actual_time'] = df['time'].str[11:]
df['actual_time'] = pd.to_datetime(df["actual_time"], format='%H:%M:%S').dt.time


# Create hour, session, actual month
def hr_func(ts):
    return ts.hour


df['hour'] = df['actual_time'].apply(hr_func)
df['hour'].astype(int)

# get the session of the hour
df['session'] = np.where(df['hour'] <= 6, 1,
                         np.where(df['hour'].between(7, 12), 2,
                                  np.where(df['hour'].between(13, 18), 3, 4)))

# get the month_year and month
df['month_year'] = df['actual_date'].dt.to_period('M')
df['month'] = df['actual_date'].dt.month
df['month'] = df['month'].astype(str)

# get weekday -- i would like  to know if there is a pattern for high retweet based on what day it was posted
df['week_day'] = df['actual_date'].dt.dayofweek

df = df[
    (df['actual_date'] >= '2016-4-1') & (df['actual_date'] <= '2016-9-30')]



# Let's clean the tweet text and create columns at the same time
def text_cleanup(column):
    # convert to lower case
    df[column] = df[column].str.lower()

    # remove URL
    df[column] = df[column].replace(r'http\S+', '', regex=True).replace(r'www\S+', '', regex=True)

    # remove under score(s) -- replace with blank
    df[column] = df[column].str.replace("_", " ")

    # extract hashtags and mention
    df['hashtag'] = df[column].str.findall(r'#.*?(?=\s|$)')
    df['mention'] = df[column].str.findall(r'@.*?(?=\s|$)')
    # create hashtag and mention count
    df['hashtag_count'] = df[column].str.count(r'#.*?(?=\s|$)')
    df['mention_count'] = df[column].str.count(r'@.*?(?=\s|$)')

    # then remove hashtags and accounts from tweets to clean the tweet text
    df[column] = df[column].str.replace(r'@.*?(?=\s|$)', " ")
    df[column] = df[column].str.replace(r'#.*?(?=\s|$)', " ")

    # remove the " character
    df[column] = df[column].str.replace('"', "")

    # # remove the \n new line
    df[column] = df[column].replace('\n', ' ', regex=True)

    df['work_count'] = df.text.str.count("work")
    df['stronger_count'] = df.text.str.count("stronger")
    df['america_count'] = df.text.str.count("america")


# execute the cleanup
text_cleanup('text')

# Now, we are ready to get the sentiment

# uncomment these two lines so you can see all the records when you print or write instead of dot dot dot ...
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

pd.options.display.max_colwidth = 350

# iterate over rows with iterrows()
list_scores = []
for index, row in df.iterrows():
    list_scores.append(sentiment_analyzer_scores(row['text']))

# make the list as dataframe
list_scores_df = pd.DataFrame(list_scores)

# join the two dataframes
df2 = pd.concat([df, list_scores_df], axis=1)

# I want to work with df dataframe :)
df = df2

# make sentiment classification (1 Positive, 0 Neutral, -1 Negative)
df['sentiment_class'] = np.where(df['compound'] >= 0.05, 1,
                                 np.where(df['compound'].between(-0.05, 0.04), 0, -1))

# now let us create the class
df['retweet_class'] = np.where(df['retweet_count'].between(1, 2000), 1,
                               np.where(df['retweet_count'].between(2001, 4000), 2,
                                        np.where(df['retweet_count'].between(4001, 6000), 3, 4)))

# favorite class
df['favorite_class'] = np.where(df['favorite_count'].between(0, 2000), 1,
                                np.where(df['favorite_count'].between(2001, 4000), 2,
                                         np.where(df['favorite_count'].between(4001, 6000), 3, 4)))

# get clinton data only to test
dfc = df[df['handle'] == 'HillaryClinton']

# get trump data only to test
dft = df[df['handle'] == 'realDonaldTrump']

# get the features for processing
dfc_tweet = dfc[['retweet_class', 'sentiment_class', 'favorite_class', 'hashtag_count', 'mention_count',
                 'week_day', 'hour', 'session', 'retweet_count', 'favorite_count',
                 'pos', 'neg', 'neu', 'compound', 'work_count', 'stronger_count', 'america_count', 'month']]

dft_tweet = dft[['retweet_class', 'sentiment_class', 'favorite_class', 'hashtag_count', 'mention_count',
                 'week_day', 'hour', 'session', 'retweet_count', 'favorite_count',
                 'pos', 'neg', 'neu', 'compound', 'work_count', 'stronger_count', 'america_count', 'month']]

# different features for experiment
# feature_names = ['sentiment_class', 'session', 'week_day', 'mention_count']
# feature_names = ['favorite_count', 'pos', 'neg', 'neu', 'session', 'week_day', 'mention_count']
# feature_names = ['compound', 'favorite_count', 'session', 'week_day', 'mention_count']
# feature_names = ['compound', 'favorite_count', 'session', 'week_day', 'mention_count', 'america_count']
feature_names = ['favorite_count', 'america_count', 'pos', 'neg', 'neu', 'session', 'week_day', 'mention_count']

# Training a Logistic Regression model with fit()
from sklearn.linear_model import LogisticRegression

# Printing the classification report
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# test clinton data only

# Separating features(X) and target(y) for retweet
X = dfc_tweet.drop('retweet_class', axis=1)
y = dfc_tweet['retweet_class']

# Splitting features and target datasets into: train and test --
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)

# # Training a Logistic Regression model with fit()
# from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train, y_train)

# Predicting the results for our test dataset
predicted_values = lr.predict(X_test)

# Printing the residuals: difference between real and predicted
# for (real, predicted) in list(zip(y_test, predicted_values)):
#     print(f'Value: {real}, pred: {predicted} {"is different" if real != predicted else ""}')

# Printing accuracy score(mean accuracy) from 0 - 1
print(f'Hillary Clinton - Accuracy score is {lr.score(X_test, y_test):.2f}/1 \n')

# # Printing the classification report
# from sklearn.metrics import classification_report, confusion_matrix, f1_score

print('Classification Report - Hillary Clinton')
print(classification_report(y_test, predicted_values))

# Printing the classification confusion matrix (diagonal is true)
print('Confusion Matrix - Hillary Clinton')
print(confusion_matrix(y_test, predicted_values))
cnf_matrix = (confusion_matrix(y_test, predicted_values))

print('Overall f1-score - Hillary Clinton')
print(f1_score(y_test, predicted_values, average="macro"))

dfc_tweet.to_csv('data/clinton_predict.csv')

# Cross validation using cross_val_score
from sklearn.model_selection import cross_val_score, ShuffleSplit

print(f'Hillary Clinton: ', cross_val_score(lr, X, y, cv=5))

# Cross validation using shuffle split
cv = ShuffleSplit(n_splits=5)
print(f'Hillary Clinton: ', cross_val_score(lr, X, y, cv=cv))

# ####
# Heat Map
sns.set()

class_names = [1, 2, 3, 4]  # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="autumn", fmt='g', center=0.00)
plt.grid()
ax.xaxis.set_label_position("bottom")
plt.title('Confusion matrix - Hillary Clinton')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
ax.set_xticklabels(class_names)
ax.set_yticklabels(class_names)
plt.tight_layout()
plt.savefig('figures/clinton-confusion-matrix.png', dpi=300)
plt.close()

# # test trump data only

# Separating features(X) and target(y) for retweet
X = dft_tweet.drop('retweet_class', axis=1)
y = dft_tweet['retweet_class']

# Splitting features and target datasets into: train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)

# # Training a Logistic Regression model with fit()
# from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train, y_train)

# Predicting the results for our test dataset
predicted_values = lr.predict(X_test)

# Printing the residuals: difference between real and predicted
# for (real, predicted) in list(zip(y_test, predicted_values)):
#     print(f'Value: {real}, pred: {predicted} {"is different" if real != predicted else ""}')

# Printing accuracy score(mean accuracy) from 0 - 1
print(f'Donald Trump - Accuracy score is {lr.score(X_test, y_test):.2f}/1 \n')

# # Printing the classification report
# from sklearn.metrics import classification_report, confusion_matrix, f1_score

print('Classification Report - Donald Trump')
print(classification_report(y_test, predicted_values))

# Printing the classification confusion matrix (diagonal is true)
print('Confusion Matrix - Donald Trump')
print(confusion_matrix(y_test, predicted_values))
cnf_matrix = (confusion_matrix(y_test, predicted_values))

print('Overall f1-score - Donald Trump')
print(f1_score(y_test, predicted_values, average="macro"))

dft_tweet.to_csv('data/trump_predict.csv')

# Cross validation using cross_val_score
from sklearn.model_selection import cross_val_score, ShuffleSplit

print(f'Donald Trump: ', cross_val_score(lr, X, y, cv=5))

# Cross validation using shuffle split
cv = ShuffleSplit(n_splits=5)
print(cross_val_score(lr, X, y, cv=cv))

# ####
# Heat Map
sns.set()

class_names = [1, 2, 3, 4]  # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="autumn", fmt='g', center=0.00)
plt.grid()
ax.xaxis.set_label_position("bottom")
plt.title('Confusion matrix - Donald Trump')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
ax.set_xticklabels(class_names)
ax.set_yticklabels(class_names)
plt.tight_layout()
plt.savefig('figures/trump-confusion-matrix.png', dpi=300)
plt.close()

# # Set 3
# dfc groupby
dfc1 = dfc.groupby(['handle', 'actual_date'], as_index=False).agg({'retweet_count': 'sum',
                                                                 'favorite_count': 'sum',
                                                                 'pos': 'sum', 'neg': 'sum', 'neu': 'sum',
                                                                 'compound': 'sum'})
# dft groupby
dft1 = dft.groupby(['handle', 'actual_date'], as_index=False).agg({'retweet_count': 'sum',
                                                                 'favorite_count': 'sum', 'pos': 'sum', 'neg': 'sum', 'neu': 'sum',
                                                                 'compound': 'sum'})

bar_width = .1
x = np.arange(len(dfc1.actual_date.unique()))


fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=100)
plt.style.use("ggplot")

b1 = ax.scatter(x, dfc1['pos'], label="Positive", alpha=0.7, color="green")

# Same thing, but offset the x by the width of the bar.
b2 = ax.scatter(x + bar_width, dfc1['neu'], label="Neutral",
                alpha=0.7, color="gray")
b3 = ax.scatter(x + bar_width, dfc1['neg'], label="Negative",
                alpha=0.7, color="red")

ax.set_xticks(x + bar_width)
ax.set_xticklabels(dfc1.actual_date.unique())

# set the ticks font to 8
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(8)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(8)

# ensure the format of the y axis using the actual values and not exponential
ax.get_yaxis().get_major_formatter().set_scientific(False)

# Add legend.
ax.legend(prop={'size': 10})

# Add axis and chart labels.
ax.set_xlabel('Tweet Date', labelpad=10, fontsize=10, fontweight='bold')
ax.set_ylabel('Sentiment Score', labelpad=10, fontsize=10, fontweight='bold')
ax.set_title('Hillary Clinton Sentiment', pad=10, fontsize=12, fontweight='bold')

fig.tight_layout()
plt.savefig('figures/hillary_sentiment.png', dpi=300)
plt.show()
plt.close()

# trump sentiment
bar_width = .1
x = np.arange(len(dft1.actual_date.unique()))


fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=100)
plt.style.use("ggplot")

b1 = ax.scatter(x, dft1['pos'], label="Positive", alpha=0.7, color="green")

# Same thing, but offset the x by the width of the bar.
b2 = ax.scatter(x + bar_width, dft1['neu'], label="Neutral",
                alpha=0.7, color="gray")
b3 = ax.scatter(x + bar_width, dft1['neg'], label="Negative",
                alpha=0.7, color="red")

ax.set_xticks(x + bar_width)
ax.set_xticklabels(dft1.actual_date.unique())

# set the ticks font to 8
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(8)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(8)

# ensure the format of the y axis using the actual values and not exponential
ax.get_yaxis().get_major_formatter().set_scientific(False)

# Add legend.
ax.legend(prop={'size': 10})

# Add axis and chart labels.
ax.set_xlabel('Tweet Date', labelpad=10, fontsize=10, fontweight='bold')
ax.set_ylabel('Sentiment Score', labelpad=10, fontsize=10, fontweight='bold')
ax.set_title('Donald Trump Sentiment', pad=10, fontsize=12, fontweight='bold')

fig.tight_layout()
plt.savefig('figures/donald_sentiment.png', dpi=300)
plt.show()
plt.close()
