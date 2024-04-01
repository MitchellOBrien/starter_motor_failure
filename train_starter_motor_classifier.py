# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 12:33:42 2024

@author: H244746
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle






file_path = os.path.join('C:\\Users\\H244746\\Desktop\\Feature Slider Example', 'starter_motor_data.xlsx')
data = pd.read_excel(file_path, sheet_name = 'Sheet1')


data = data.sort_values(by = ['failure', 'start_time'])
data = data.reset_index(drop = 'True')


X = data['start_time'].values.reshape(-1, 1)
y = data['failure']


data['start_time'] = data['start_time'] + np.random.normal(5, 10, len(data['start_time']))


model = LogisticRegression()
model.fit(X, y)

X.values.reshape(-1,1)

pred = model.predict(X)

(pred == y).sum() / len(y)


model_path = 'C:\\Users\\H244746\\Desktop\\Feature Slider Example\\model\\starter_motor_classifier.sav'


with open(model_path, 'wb') as handle:
    pickle.dump(model, handle)

    "    performance_path = 'C:\\\\Users\\\\H244746\\\\Desktop\\\\Deployment Exercise\\\\performance_metrics\\\\loan_default_classifier_performance_test.sav'\n",
    "    with open(performance_path, 'wb') as handle:\n",
    "        pickle.dump(performance_metrics, handle)\n",

