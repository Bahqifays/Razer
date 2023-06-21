"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.7
"""

import pandas as pd
import numpy as np
import re

# Define a function to transform column names
def transform_column_name(col):
    # Convert capital letter to small letter
    col = col.lower()
    # Replace space with underscore
    col = re.sub(' ', '_', col)
    # Remove leading and trailing special characters
    col = re.sub('^[^a-zA-Z0-9]*|[^a-zA-Z0-9]*$', '', col)
    # Replace special characters with underscore
    col = re.sub('[^a-zA-Z0-9]+', '_', col)
    return col

def preprocess_razer(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data of Razer transactions

    Args:
        raw_data: raw data of Razer transactions from portal
    Returns:
        Preprocessed data, with columns renamed and success payment status selected
    """

    # Select on success payment status
    razer_success = raw_data[raw_data['Status'] == 'SUCCESS']

    # Apply column rename function to the DataFrame
    new_columns = {old_col: transform_column_name(old_col) for old_col in razer_success.columns}
    razer_success = razer_success.rename(columns=new_columns)

    # Create new 'payment_date_only' column
    razer_success['payment_date_only'] = pd.to_datetime(razer_success['payment_date'])

    # Remove the time portion while maintaining the datetime type
    razer_success['payment_date_only'] = razer_success['payment_date_only'].dt.floor('D')

    # Split the 'amount' column into two columns
    razer_success[['currency', 'txn_amount']] = razer_success.loc[:, ('amount')].str.split(' ', n=1, expand=True)

    # Convert the 'txn_amount' column to float datatype
    # replace comma with empty if there is any (eg: 1,000 to 1000)
    razer_success['txn_amount'] = razer_success['txn_amount'].apply(lambda x: x.replace(',', '')).astype(float)

    # Convert 'channel' to lowercase and replace spaces with underscores
    razer_success['channel_copy'] = razer_success['channel'].apply(lambda x: x.lower().replace(' ', '_').replace('-', '_'))

    return razer_success


def calculation_razer(preprocessed_data: pd.DataFrame, channels: dict) -> pd.DataFrame:
    """Calculate transaction charges of Razer

    Args:
        preprocessed_data: cleansed data of Razer transactions
        channels: dictionary of rate of each channel spelt out in parameters.yml
    Returns:
        Calculated transactions data
    """

    razer_calculated = preprocessed_data.copy()

    # Create a dictionary of lists
    channel_dict = {key: list(value.keys()) for key, value in channels.items()}

    # Create the new 'rate_type' column based on the 'channel' column
    razer_calculated['rate_type'] = razer_calculated['channel_copy'].apply(lambda x: next(key for key, value in channel_dict.items() if x in value))

    # Create a new column containing the values based on the keys in the 'channel' column
    razer_calculated['rate'] = razer_calculated['channel_copy'].map({k: v for d in channels.values() for k, v in d.items()})
    
    # Create the 'txn_charge' column based on the conditions of 'rate_type'
    razer_calculated['txn_charge'] = np.where(razer_calculated['rate_type'] == 'fixed', razer_calculated['rate'], 
                                              razer_calculated['txn_amount'] * razer_calculated['rate'])
    
    # Calculate net_amount
    razer_calculated['net_amount'] = razer_calculated['txn_amount'] - razer_calculated['txn_charge']
    
    # Select only the relevant columns as the final dataset
    final_cols = ['payment_date', 'order_id', 'txn_id', 'status', 'type', 'channel',
                  'rate_type', 'rate', 'currency', 'txn_amount', 'txn_charge', 
                  'net_amount']
    razer_final = razer_calculated[final_cols]
    
    return razer_final