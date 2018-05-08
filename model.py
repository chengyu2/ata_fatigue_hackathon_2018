from typing import Dict, Tuple
import os
import tensorflow as tf
from tensorflow.python.estimator.canned.dnn_linear_combined import DNNLinearCombinedClassifier

from global_config import GlobalVars
from input_pipeline import csv_input_fn, label_dict
from preprocessing import format_column_name
from utils.file_util import FileUtil

dense_columns = [tf.feature_column.embedding_column(
    tf.feature_column.bucketized_column(
        tf.feature_column.numeric_column(format_column_name("Event_time_(local)"), dtype=tf.int32), boundaries=[8, 16]), dimension=GlobalVars.emb_dim),

    tf.feature_column.numeric_column(format_column_name("Duration_(sec)")),
    tf.feature_column.numeric_column(format_column_name("Speed_(mph)")),
    tf.feature_column.numeric_column(format_column_name("Distance_covered (metres)")),
    tf.feature_column.numeric_column(format_column_name("Bearing_(degrees)")),

    tf.feature_column.numeric_column(format_column_name("Time_into_trip_(minutes)")),
    tf.feature_column.numeric_column(format_column_name("Trip_distance_(miles)"))]

categorical_columns = [tf.feature_column.categorical_column_with_vocabulary_list(
    key="GPIO_alert",
    vocabulary_list=["Yes", "No"]),
    tf.feature_column.categorical_column_with_vocabulary_list(
        key="Audio_alert",
        vocabulary_list=["Yes", "No"]),
    tf.feature_column.categorical_column_with_vocabulary_list(
        key="Type",
        vocabulary_list=["distraction", "fatigue"]),
    tf.feature_column.categorical_column_with_vocabulary_list(
        key="Detected_type",
        vocabulary_list=["glance_down", "glance_away", "fatigue"]),
]

estimator = DNNLinearCombinedClassifier(
    # wide settings
    linear_feature_columns=categorical_columns + dense_columns,
    linear_optimizer=tf.train.FtrlOptimizer(learning_rate=0.001, l1_regularization_strength=1e-4,
                                            l2_regularization_strength=1e-4),
    # deep settings
    dnn_feature_columns=[tf.feature_column.embedding_column(column, dimension=GlobalVars.emb_dim) for column in
                         categorical_columns],
    dnn_hidden_units=[128, 64, 32],
    dnn_optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.001, l1_regularization_strength=1e-4,
                                                    l2_regularization_strength=1e-4),
    # warm-start settings
    model_dir=GlobalVars.model_dir,
    n_classes=len(label_dict)
)


def input_fn_train() -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:  # returns x, y
    return csv_input_fn(csv_path=GlobalVars.train_path, batch_size=GlobalVars.batch_size)




# Eval (Evaluate the trained model using a separate dataset not available during training)
def input_fn_eval() -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:  # returns x, y
    return csv_input_fn(csv_path=GlobalVars.val_path, batch_size=GlobalVars.batch_size)

FileUtil.clear_folder(folder_path=GlobalVars.model_dir)

for _ in range(100):
    estimator.train(input_fn=input_fn_train, steps=100)

    metrics = estimator.evaluate(input_fn=input_fn_eval)


# Prediction (Unlabelled data)
def input_fn_predict():  # returns x, None
    pass
# predictions = estimator.predict(input_fn=input_fn_predict)
