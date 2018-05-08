from typing import List, Dict

import pandas as pd
import numpy as np
from global_config import GlobalVars
from datetime import datetime


def preprocess_data() -> (List[str], Dict[str, int]):
    label_dict = {}
    df = pd.read_csv(filepath_or_buffer=GlobalVars.raw_csv_path)

    print(df.head(5))
    print(df.columns)
    df = format_column_names(df=df)
    event_time_local_idx = df.columns.get_loc(format_column_name("Event time (local)"))
    label_idx = df.columns.get_loc(format_column_name("Sub-type"))

    for i in range(df.shape[0]):
        df.iat[i, event_time_local_idx] = datetime.strptime(df.iloc[i, event_time_local_idx], '%d/%m/%y %H:%M' ).hour

        # Change string label name into label indices tf.int32
        label_name = df.iloc[i, label_idx]
        if label_name not in label_dict:
            label_dict[label_name] = len(label_dict)
        df.iat[i, label_idx] = label_dict[label_name]


    print(df[format_column_name("Event time (local)")])



    df_train, df_val = np.split(df, [int(df.shape[0]*0.9)], axis=0)
    df_train.to_csv(path_or_buf=GlobalVars.train_path, index=False)
    df_val.to_csv(path_or_buf=GlobalVars.val_path, index=False)
    return list(df.columns), label_dict



def format_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.rename(columns=lambda x: format_column_name(x), inplace=True)
    return df

def format_column_name(col_name: str) -> str:
    return col_name.replace(" ", "_").replace("(", "/").replace(")", "/")
