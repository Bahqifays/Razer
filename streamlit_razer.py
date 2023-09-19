import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re

def load_data(dir):
    df = pd.read_excel(dir)
    return df

# Function to replace NaN values with "N/A" for display
def display_df(df):
    df_display = df.copy()
    df_display = df_display.applymap(lambda x: 'N/A' if pd.isna(x) else x)
    return df_display

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

    # Create new 'payment_date' column
    razer_success['payment_date'] = pd.to_datetime(razer_success['payment_date'])

    # Sort the DataFrame by the datetime column from earliest to newest
    razer_success = razer_success.sort_values(by='payment_date')

    # Remove the time portion while maintaining the datetime type
    razer_success['payment_date'] = razer_success['payment_date'].dt.floor('D')

    # Split the 'amount' column into two columns
    razer_success[['currency', 'txn_amount']] = razer_success.loc[:, ('amount')].str.split(' ', n=1, expand=True)

    # Convert the 'txn_amount' column to float datatype
    # replace comma with empty if there is any (eg: 1,000 to 1000)
    razer_success['txn_amount'] = razer_success['txn_amount'].apply(lambda x: x.replace(',', '')).astype(float)

    # Convert 'channel' to lowercase and replace spaces with underscores
    razer_success['channel_copy'] = razer_success['channel'].apply(lambda x: x.lower().replace(' ', '_').replace('-', '_'))

    # final_cols = ['payment_date', 'order_id', 'txn_id', 'status', 'type', 'channel',
    #               'currency', 'txn_amount']
    # razer_success = razer_success[final_cols]

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

        # Calculate the sum for the last 3 columns
    total_row = razer_final.iloc[:, -3:].sum()

    # Append the total row to the DataFrame
    razer_final = razer_final.append(total_row, ignore_index=True)
    
    return razer_final

st.divider()
st.title('Razer Settlement')
st.divider()

title = st.text_input('File Directory:', 
                      r"C:\Users\syafiq\OneDrive - Malaysian Rating Corporation Berhad\MARC Data\Kedro\Razer\razer-settlement\data\01_raw\payment_report_1694663384.xlsx")

df = load_data(title)
df_razer = preprocess_razer(df)

final_cols = ['payment_date', 'order_id', 'txn_id', 'status', 'type', 'channel',
              'currency', 'txn_amount']
df_razer_success = df_razer[final_cols]
df_razer_success = df_razer_success.reset_index(drop=True)
df_display_main = display_df(df_razer_success)

st.write("This is the raw Razer transactions report:")
st.dataframe(df_display_main)

# Add a date range selector
start_date = st.date_input('Start Date', min_value=min(df_razer['payment_date']), max_value=max(df_razer['payment_date']), 
                           value=min(df_razer['payment_date']))
end_date = st.date_input('End Date', min_value=min(df_razer['payment_date']), max_value=max(df_razer['payment_date']), 
                         value=max(df_razer['payment_date']))

# Convert the date column to Pandas datetime
df_razer['payment_date'] = pd.to_datetime(df_razer['payment_date']).dt.date

# Use st.cache to cache the data filtering
@st.cache_data
def filter_data(df, start_date, end_date):
    return df[(df['payment_date'] >= start_date) & (df['payment_date'] <= end_date)]

filtered_df = filter_data(df_razer, start_date, end_date)

# Group the DataFrame by 'channel' and calculate the total amount
channel_total = filtered_df.groupby('channel')['txn_amount'].sum().reset_index()

# Sort by decreasing total amount
channel_total = channel_total.sort_values(by='txn_amount', ascending=False)

# Create a horizontal bar chart using Plotly Express
st.write('Total Amount by Channel')

fig = px.bar(channel_total, x='txn_amount', y='channel', orientation='h',
             labels={'txn_amount': 'Total Amount'}, color='channel',
             text='txn_amount', color_discrete_sequence=px.colors.qualitative.Set1)

# Customize the chart appearance
fig.update_traces(textposition='outside')

st.plotly_chart(fig)

channels = {'variable': {'credit': 0.0135, 'boost': 0.013, 'wechatpaymy': 0.01, 'grabpay': 0.013, 'mb2u_qrpay_push': 0.01,
                         'tng_ewallet': 0.013, 'shopeepay': 0.013, 'rpp_duitnowqr': 0.0095, 'alipay': 0.013, 'alipayplus': 0.013,
                         'unionpay': 0.023, 'fpx': 0.012, 'fpx_b2b': 0.012, 'atome': 0.05},
            'fixed': {'fpx_mb2u': 1, 'mb2u': 1, 'fpx_cimbclicks': 1, 'cimb_clicks': 1, 'fpx_rhb': 1, 'rhb_onl': 1, 'fpx_pbb': 1,
                      'pbebank': 1, 'fpx_hlb': 1, 'hlb_onl': 1, 'fpx_bimb': 1, 'fpx_amb': 1, 'amb_w2w': 1, 'fpx_abmb': 1, 
                      'alb_onl': 1, 'fpx_abb': 1, 'affin_epg': 1, 'fpx_bmmb': 1, 'fpx_bkrm': 1, 'fpx_bsn': 1, 'fpx_ocbc': 1,
                      'fpx_uob': 1, 'fpx_hsbc': 1, 'fpx_scb': 1, 'fpx_kfh': 1, 'fpx_b2b_amb': 1.4, 'fpx_b2b_hlb': 1.4,
                      'fpx_b2b_uob': 1.4, 'fpx_b2b_abb': 1.4, 'fpx_b2b_hsbc': 1.4, 'fpx_m2e': 1.4, 'fpx_b2b_cimb': 1.4,
                      'fpx_b2b_bimb': 1.4, 'fpx_b2b_rhb': 1.4, 'fpx_b2b_pbb': 1.4, 'fpx_b2b_kfh': 1.4, 'fpx_b2b_deutsche': 1.4,
                      'fpx_b2b_abmb': 1.4, 'fpx_b2b_scb': 1.4, 'fpx_b2b_ocbc': 1.4, 'fpx_b2b_bmmb': 1.4, 'fpx_emandate': 1.4,
                      'fpx_directdebit': 1.4, 'fpx_agrobank': 1, 'fpx_b2b_agrobank': 1.4, 'fpx_b2b_abbm': 1.4, 'fpx_b2b_citibank': 1.4,
                      'fpx_b2b_bkrm': 1.4, 'fpx_b2b_pbbe': 1.4, 'fpx_b2b_uobr': 1.4, 'fpx_emandate_abb': 1.4, 'fpx_directdebit_abb': 1.4}
            }

df_razer_settlement = calculation_razer(filtered_df, channels)
st.dataframe(df_razer_settlement)