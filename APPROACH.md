1. **Plan an Approach**
What steps do we plan to follow and why?

To tackle this project, I followed a structured approach to build a forecasting model in the most efficient way possible, given the time constraints. The key steps included preparing the data, creating useful features, choosing the right model, training it, and making predictions. My goal was to create a model that could provide accurate forecasts while balancing speed and efficiency.

For those who may not be familiar with data science, the idea here is to create a model that given some characteristics from the past, it will "learn" the patterns and then it can give us a predicition for future sales.

2. **Steps I Followed**

***Data Preparation***

This part I used process.py as recommend and after that I was able to do a very small feature engineering (could be much better, if there was a time for that).

***Feature Engineering***

This part was to create some new features and the most important structure from my modeling (28 day lags), I decided to use this pattern since I am modeling a forecasting problem as a regression.

***Choosing and Training the Model***

I selected LightGBM because it is fast and handles large datasets well. It also has important characteristics, like: handle missing values and categorical features, which gives me enough velocity to run it fast.

I used an automated tuning process (Optuna) to adjust the model’s settings for better performance. Optuna is well known package and it has statistical approach to tuning the hyperparams in a way that overfitting is prevented.

Then, optuna found the best set of hyperparams that were used to train the model and saved it as pkl format.

***Making Predictions and Model in Production***

In order to make the predicitons I had to create the process_submission.py, which was necessary to create the dataset in the same format of the trainning phase.

After that, it is necessary to transform the data using the same steps used in the trainning phase and then the predicitons were generated.

All the important functions created were located at helper.py.

The submission.csv contains all the predicitions.

In order to evaluate the model I used RSME on train ans validation datasets, as regular way to evaluate that.

The final RSME could be extracted by comparing the test results and the actual ones.

In production this is difficult, since we don't have ground truth. The alternative is create some methods to evaluate data drift and concept drift, once it is happening, we need to train the model again or re-design the entire conception of the model.

***What Could Be Improved***

Since time was limited, I couldn’t dive deep into exploratory data analysis or refine the features as much as I would have liked. The code could also be cleaned up to follow best practices (PEP8) and be better structured using tools like MLflow or pipelines. Also, I would document the jupyter notebooks better.

Also change the approach to have more than one model, for instance: the items that have more sales are not well suited in this model, then create a new one to those items would be a great improvement in the metrics.