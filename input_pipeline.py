import tensorflow as tf
import os
from preprocessing import preprocess_data

UNK = "Unknown"
COLUMNS, label_dict = preprocess_data()
print("{} columns: {}".format(len(COLUMNS), COLUMNS))
print("Labels: {}".format(label_dict))

FIELD_DEFAULTS = [[UNK], [UNK], [UNK], [0], [0], [UNK],
                  [UNK], [0.0], [0.0], [0.0], [0.0],
                  [0], [UNK], [len(label_dict)], [UNK], [UNK],
                  [0.0], ["No"], ["No"], [UNK], [0.00],
                  [0.00], [0.00]]

assert len(COLUMNS) == len(FIELD_DEFAULTS)

def _parse_line(line):
    # Decode the line into its fields
    fields = tf.decode_csv(line, FIELD_DEFAULTS)

    # Pack the result into a dictionary
    features = dict(zip(COLUMNS, fields))

    # Separate the label from the features
    label_col_name = "Sub-type"

    label = features.pop(label_col_name)

    return features, label

def csv_input_fn(csv_path: str, batch_size: int):
    # Create a dataset containing the text lines.
    dataset = tf.data.TextLineDataset(csv_path).skip(1)

    # Parse each line.
    dataset = dataset.map(_parse_line, num_parallel_calls=os.cpu_count())

    # Shuffle, repeat, and batch the examples.
    if "train" in csv_path:
        dataset = dataset.shuffle(1000).repeat(5).batch(batch_size)
    else:
        dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset.make_one_shot_iterator().get_next()
