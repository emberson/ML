# ------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------

This repository contains examples of machine learning projects I have been working on mainly for my
own learning experience and personal pleasure. All programs are written in Python 3. See specific
documentation below for each project. 

# ------------------------------------------------------
# PREDICTING NHL PLAYOFF SERIES WINNERS
# ------------------------------------------------------

This project grew out of my desire to determine how well NHL playoff series can be predicted using
machine learning applications. Going into this, I was not overly optimistic about the accuracy one
can obtain based on previous work in the literature done in the field of sports predictions. The
main difficulty with sports predictions is that a large fraction of the game can be attributed to
random events which reduce predictive power. Previous works have shown that this induces a ceiling
that defines the maximum accuracy at which a single game can be predicted using statistical data.
This is particularly true in the case of hockey since scoring events are relatively low and are
thus easily overwhelmed by random noise in the game. Nevertheless, it is worth testing how well
machine learning applications can predict NHL playoff series success especially since hockey has
been largely unexplored in comparison to other sports. In fact, I could find only one other work
[1] that focused on hockey predictions. This work found that a single NHL game can be predicted
at an accuracy of about 60%; it is reasonable to expect a similar level of accuracy for playoff
series predictions. In addition to gauging playoff prediction capabilities, I was interested in
determining which team attributes are most correlated with playoff success. 

The first step in this project was to gather regular season and playoff series stats from previous 
NHL seasons. This was achieved using a combination of data from https://www.hockey-reference.com/
and http://naturalstattrick.com. For each playoff series, I recorded which of the two teams won
and whether the winning team had home series advantage or not. In addition, I recorded various
stats for each of the two teams based on their performance in the regular season. Traditional
stats include average per-game points, goals, and shots as well as advanced statistics such
as Fenwick (unblocked shot attempts), Corsi (all shot attempts), and PDO (sum of shooting and saving
percentage). This data was then grouped into three categories: (1) results from all regular season
games against all teams; (2) results from regular season matchups between the two playoff teams
(to gauge how well the two teams performed only against each other); (3) results from the last N
games of the regular season (to gauge recent performance and account for injuries, trades, etc.)
Here N is a hyperparameter of the model which I took to be N = 7 for now. This data was collected
by writing a web crawler that used the BeautifulSoup library to feed data into Pandas dataframes.

The dataset is relatively small covering the past 11 NHL seasons that sum to a total of 165
playoff series matchups. The data bottleneck is simply due to the non-availability of regular
season data prior to the 2007 NHL season. The data was trained on a variety of classifiers including
Logistic Regression, Support Vector Machine, Naive Bayes, Random Forest, and a Neural Network. The
Neural Network was built using Keras while the others used scikit-learn. The data was split 70-30
between training and testing with five-fold cross validation used to tune each model. The baseline 
to beat is based on picking the team with home series advantage to win. I found that the home series
team has a minor edge in playoff series success with 54% of the series wins in the training set. 
The most success was obtained using the Naive Bayes model and the Neural Network. These achieved
60% and 62% accuracy, respectively, and differ from the baseline by 1 sigma. The other models 
obtained <60% accuracy and were not significantly different from the baseline. 

The main challenge with this project was the large variability in the data. This manifest as a 
large scatter in the 5-fold cross validation accuracies. This is likely attributed to both the
small sample size as well as the high degree of randomness intrinsic to hockey matches. The model
would likely benefit from an increased sample size though it is doubtful to expect accuracies 
much larger than those found here. This is based on single-game predictions studied in previous
works that found similar levels of accuracy, but were not as data limited. Through this analysis,
I found that the most predictive attributes were goal differential, Fenwick differential, and
point differential; each of these being averaged over all regular season games. Interestingly,
the next three most predictive variables were those same three attributes, but based on averages
collected over the last N games. Somewhat surprisingly, the data collected from regular season
matchups between the two playoff teams was not very predictive. The reason may be due to the fact
that some of the teams only played a small number (<= 3) of regular season games against each other
and thus are more susceptible to noise. 

# REFERENCES
[1] Weissbock, J. Forecasting Success in the National Hockey League using In-Game Statistics 
    and Textual Data. Masters Thesis. University of Ottawa. 2014. 

