# -*- coding:utf-8 -*-
# Created by shellbye on 2018/8/2.
import tensorflow as tf

import os
import sys

from IPython.display import clear_output


models_path = os.path.join(os.getcwd(), 'models')

sys.path.append(models_path)

from official.wide_deep import census_dataset

census_dataset.download("/tmp/census_data/")

# export PYTHONPATH=${PYTHONPATH}:"$(pwd)/models"
# running from python you need to set the `os.environ` or the subprocess will not see the directory.

if "PYTHONPATH" in os.environ:
    os.environ['PYTHONPATH'] += os.pathsep + models_path
else:
    os.environ['PYTHONPATH'] = models_path

train_file = "/tmp/census_data/adult.data"
test_file = "/tmp/census_data/adult.test"

import pandas

train_df = pandas.read_csv(train_file, header=None, names=census_dataset._CSV_COLUMNS)
test_df = pandas.read_csv(test_file, header=None, names=census_dataset._CSV_COLUMNS)

train_df.head()


def easy_input_function(df, label_key, num_epochs, shuffle, batch_size):
    label = df[label_key]
    ds = tf.data.Dataset.from_tensor_slices((dict(df), label))

    if shuffle:
        ds = ds.shuffle(10000)

    ds = ds.batch(batch_size).repeat(num_epochs)

    return ds


# ds = easy_input_function(train_df, label_key='income_bracket', num_epochs=5, shuffle=True, batch_size=10)
#
# for feature_batch, label_batch in ds.take(1):
#   print('Some feature keys:', list(feature_batch.keys())[:5])
#   print()
#   print('A batch of Ages  :', feature_batch['age'])
#   print()
#   print('A batch of Labels:', label_batch )

# import inspect
# print(inspect.getsource(census_dataset.input_fn))

ds = census_dataset.input_fn(train_file, num_epochs=5, shuffle=True, batch_size=10)

# 这里必须要初始化，才能在下面get_next，前提还得disable掉eager_execution
# https://stackoverflow.com/questions/50285097/tensorflow-tf-enable-eager-execution-must-be-called-at-program-startup
# https://tensorflow.google.cn/programmers_guide/datasets
iterator = ds.make_initializable_iterator()
feature_batch, label_batch = iterator.get_next()
# for feature_batch, label_batch in ds.take(1):

print('Feature keys:', list(feature_batch.keys())[:5])
print()
print('Age batch   :', feature_batch['age'])
print()
print('Label batch :', label_batch)

import functools

train_inpf = functools.partial(census_dataset.input_fn, train_file, num_epochs=2, shuffle=True, batch_size=64)
test_inpf = functools.partial(census_dataset.input_fn, test_file, num_epochs=1, shuffle=False, batch_size=64)

fc = tf.feature_column
################################################################################
# 只用年龄这一个特征
age = tf.feature_column.numeric_column('age')
# t = tf.feature_column.input_layer(feature_batch, [age])
classifier = tf.estimator.LinearClassifier(feature_columns=[age])
classifier.train(train_inpf)
result = classifier.evaluate(test_inpf)
clear_output()  # used for display in notebook
for key, value in sorted(result.items()):
    print('%s: %s' % (key, value))
################################################################################
# 结合其他特征
education_num = tf.feature_column.numeric_column('education_num')
capital_gain = tf.feature_column.numeric_column('capital_gain')
capital_loss = tf.feature_column.numeric_column('capital_loss')
hours_per_week = tf.feature_column.numeric_column('hours_per_week')
my_numeric_columns = [age, education_num, capital_gain, capital_loss, hours_per_week]
# t1 = tf.feature_column.input_layer(feature_batch, my_numeric_columns)
classifier = tf.estimator.LinearClassifier(feature_columns=my_numeric_columns)
classifier.train(train_inpf)
result = classifier.evaluate(test_inpf)
clear_output()
for key, value in sorted(result.items()):
    print('%s: %s' % (key, value))
################################################################################
# 其他特征
relationship = tf.feature_column.categorical_column_with_vocabulary_list(
    'relationship',
    ['Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried', 'Other-relative'])
occupation = tf.feature_column.categorical_column_with_hash_bucket(
    'occupation', hash_bucket_size=1000)
education = tf.feature_column.categorical_column_with_vocabulary_list(
    'education', [
        'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
        'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
        '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])
marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
    'marital_status', [
        'Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
        'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'])
workclass = tf.feature_column.categorical_column_with_vocabulary_list(
    'workclass', [
        'Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
        'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'])
my_categorical_columns = [relationship, occupation, education, marital_status, workclass]
classifier = tf.estimator.LinearClassifier(feature_columns=my_numeric_columns + my_categorical_columns)
classifier.train(train_inpf)
result = classifier.evaluate(test_inpf)
clear_output()
for key, value in sorted(result.items()):
    print('%s: %s' % (key, value))
################################################################################

import tempfile
age_buckets = tf.feature_column.bucketized_column(
    age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
base_columns = [
    education, marital_status, relationship, workclass, occupation,
    age_buckets,
]
crossed_columns = [
    tf.feature_column.crossed_column(
        ['education', 'occupation'], hash_bucket_size=1000),
    tf.feature_column.crossed_column(
        [age_buckets, 'education', 'occupation'], hash_bucket_size=1000),
]
classifier = tf.estimator.LinearClassifier(
    model_dir=tempfile.mkdtemp(),
    feature_columns=base_columns + crossed_columns,
    optimizer=tf.train.FtrlOptimizer(learning_rate=0.1))
classifier.train(train_inpf)
results = classifier.evaluate(test_inpf)
clear_output()
for key,value in sorted(result.items()):
  print('%s: %0.2f' % (key, value))
################################################################################
import numpy as np

predict_df = test_df[:20].copy()

pred_iter = classifier.predict(
    lambda:easy_input_function(predict_df, label_key='income_bracket',
                               num_epochs=1, shuffle=False, batch_size=10))

classes = np.array(['<=50K', '>50K'])
pred_class_id = []

for pred_dict in pred_iter:
  pred_class_id.append(pred_dict['class_ids'])

predict_df['predicted_class'] = classes[np.array(pred_class_id)]
predict_df['correct'] = predict_df['predicted_class'] == predict_df['income_bracket']

clear_output()

print(predict_df[['income_bracket','predicted_class', 'correct']])
