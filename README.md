# Event-Fraud-Detection
This was a project that was given during my time (2018) at Galvanize found here:
https://github.com/gSchool/dsi-fraud-detection-case-study

We were given a REST api on a heroku server and we had to run our model against new events to detect potential cases of fraud. We ran a gradient boosting classifier to measure the potential risk and categorized the probability into three buckets: Low Risk, Medium Risk, High Risk.


This project includes a mongodb database where we stored our updated predictions and casted them on a flask app.


