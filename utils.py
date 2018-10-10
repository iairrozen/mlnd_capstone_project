# Auxiliary functions

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from IPython.display import Markdown, display

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             fbeta_score, classification_report)

plt.style.use('fivethirtyeight')

seed = 42


# EDA UTILS ##################################################################

# ----------------------------------------------------------------------------
# Print markdown programatically for better visualization.
def printmd(string):
    display(Markdown(string))


# ----------------------------------------------------------------------------
# Plots and creates labels for values plotted. Used for exploration of features.
def plot_and_annotate(data, **kwarg):
    ax = data['percent'].sort_values().plot(kind='barh', **kwarg)
    for p in ax.patches:
        width = p.get_width()
        plt.text(5 + width, p.get_y() + 0.55 * p.get_height(),
                 '{:1.2f}%'.format(width),
                 ha='center', va='center')


# ----------------------------------------------------------------------------
# Creates percentages for value_counts of a pd.series object.
def create_count_percentages(series, name=None):
    values = series.value_counts()
    expanded_value = values.to_frame(name)
    expanded_value['percent'] = values.apply(lambda _: _ / series.size * 100)
    return expanded_value


# ----------------------------------------------------------------------------
# Data must be loaded before definin create_plot_data due to inside reference.
def create_plot_data(data, feature):
    return create_count_percentages(data[feature], name=feature)


# ----------------------------------------------------------------------------
def cast_df_features(data, feature_catalog):
    def parse_type(dtype):
        if dtype == 'int':
            return np.int8
        elif dtype == 'float':
            return np.float
        else:
            return dtype

    # Make a dict to use as dtypes for panda's dataframe
    features_dtypes = feature_catalog.set_index('feature_name')['pandas_dtype'].apply(parse_type).to_dict()
    # Keep only the columns that remain in the clean version of the dataframe
    features_dtypes = {k: v for k, v in features_dtypes.items() if k in data.columns}
    return data.astype(features_dtypes)

# EDA UTILS ##################################################################


# CLASSIFICATION UTILS #######################################################

def create_one_hot_encoding(data, feature_catalog):
    categorical_features = list(feature_catalog[feature_catalog['feature_type'] == 'categorical']['feature_name'])
    categorical_features.remove('crashSeverity')
    categorical_features = list(filter(lambda f: f in data.columns, categorical_features))
    return pd.get_dummies(data, columns=categorical_features)


# ----------------------------------------------------------------------------
def structure_and_print_results(model_name, dataset_variation, y_true, y_pred, betta=1, digits=2, average=None):
    precision = precision_score(y_true, y_pred, average=average)
    recall = recall_score(y_true, y_pred, average=average)
    f1 = f1_score(y_true, y_pred, average=average)
    print(classification_report(y_true, y_pred, digits=digits))

    n_classes = len(precision)
    frame_data = {
        'Model': [model_name] * n_classes,
        'Variation': [dataset_variation] * n_classes,
        'Target': ['F', 'M', 'N', 'S'],
        'Precision': precision,
        'Recall': recall,
        'F1_score': f1
    }
    return pd.DataFrame(frame_data)

##############################################################################
