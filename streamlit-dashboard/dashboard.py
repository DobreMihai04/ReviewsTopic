import streamlit as st
import pandas as pd
import plotly.express as px
import boto3
import pyarrow.parquet as pq
import pyarrow as pa
from datetime import datetime
from io import BytesIO
from st_aggrid import AgGrid, GridOptionsBuilder
import os


# AWS credentials and bucket details
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
BUCKET_NAME = os.getenv("BUCKET_NAME")
BASE_PATH = os.getenv("BASE_PATH")


# Streamlit Page Configurations
st.set_page_config(layout='wide', initial_sidebar_state='expanded')
st.markdown("<h1 style='text-align: center;'>Review Analysis Dashboard</h1>", unsafe_allow_html=True)

# === Utility Functions ===
def initialize_s3_client():
    """Initialize S3 client using credentials."""
    return boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

def load_data_from_s3(s3_client, bucket, directories):
    """Load data from S3 bucket."""
    tables = {}
    for directory in directories:
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=f"{BASE_PATH}{directory}/")
        parquet_files = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.parquet')]
        
        data_frames = []
        for file_key in parquet_files:
            file_response = s3_client.get_object(Bucket=bucket, Key=file_key)
            file_content = file_response['Body'].read()
            table = pq.read_table(BytesIO(file_content))
            data_frames.append(table)

        if data_frames:
            tables[directory] = pa.concat_tables(data_frames).to_pandas().reset_index(drop=True)
    return tables

def plot_line_chart(df, x_col, y_col, title, x_label, y_label):
    fig = px.line(df, x=x_col, y=y_col, title=title, labels={x_col: x_label, y_col: y_label})
    fig.update_layout(
        plot_bgcolor='rgba(245,245,245,1)',  
        paper_bgcolor='rgba(245,245,245,1)',  
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title=dict(text=x_label, font=dict(size=18, family='Arial', color='grey')), 
        yaxis_title=dict(text=y_label, font=dict(size=16, family='Arial', color='grey'))  
    )
    return fig

def plot_bar_chart(df, x_col, y_col, title, x_label, y_label, hover_name, color=None, y_range=None):
    """Create a Plotly bar chart with a grey background."""
    fig = px.bar(df, x=x_col, y=y_col, color=color, title=title, labels={x_col: x_label, y_col: y_label}, hover_name=hover_name)
    if y_range:
        fig.update_yaxes(range=y_range)
    fig.update_layout(
        plot_bgcolor='rgba(245,245,245,1)',  
        paper_bgcolor='rgba(245,245,245,1)', 
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title=dict(text=x_label, font=dict(size=18, family='Arial', color='grey')), 
        yaxis_title=dict(text=y_label, font=dict(size=16, family='Arial', color='grey'))   
    )
    return fig

def filter_data_by_date(df, date_col, start_date, end_date):
    """Filter data based on date range."""
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    return df[(df[date_col] >= start_date) & (df[date_col] <= end_date)]

def prepare_and_sort(df, group_col, sort_col, ascending=False, top_n=10):
    """Group, sort, and return top N rows."""
    grouped = df.groupby(group_col)[sort_col].sum().reset_index()
    return grouped.sort_values(by=sort_col, ascending=ascending).head(top_n)

def prepare_score_data(filtered_topic_data, min_reviews=2, top_n=10):
    """Prepare data for highest and lowest average scores."""
    high_score_avg = filtered_topic_data.groupby('topic_name').agg(
        average_score=('average', 'mean'),
        total_count=('count', 'sum')
    ).reset_index()
    
    high_score_avg = high_score_avg[high_score_avg['total_count'] > min_reviews]
    high_score_avg['short_name'] = high_score_avg['topic_name'].apply(lambda x: x if len(x) <= 20 else x[:20] + '...')
    
    return high_score_avg

def render_top_cards(df, num_cards, title):
    """Render top N topics in card format."""
    cols = st.columns(num_cards)
    for index, (col, row) in enumerate(zip(cols, df.itertuples())):
        col.markdown(f"""
        <div style="display: flex; flex-direction: column; justify-content: space-between; align-items: center; padding: 10px; border: 1px solid #ddd; border-radius: 5px; margin-bottom: 10px; height: 150px;">
            <div style="flex-grow: 1; text-align: center; margin-bottom: auto;">
                <h3 style="margin: 0; font-size: 16px;">{row.topic_name[:15]}...</h3>
            </div>
            <div style="flex-grow: 1; font-size: 24px; font-weight: bold; text-align: center; margin: auto 0;">
                {round(row.average, 2)}
            </div>
            <div style="flex-grow: 1; text-align: center; margin-top: auto; color: {'red' if row.score_change < 0 else 'green'};">
                {abs(round(row.score_change, 2))} change since last month
            </div>
        </div>
        """, unsafe_allow_html=True)

# === Load Data from S3 ===
s3_client = initialize_s3_client()
DIRECTORIES = [
    'total_aggregations', 'top_topics_by_score', 'scores_count',
    'agg_by_date', 'agg_by_month', 'agg_by_month_topic',
    'agg_by_week', 'agg_by_week_topic', 'raw_data', 'agg_by_date_topic'
]
tables = load_data_from_s3(s3_client, BUCKET_NAME, DIRECTORIES)

# Unpack DataFrames
agg_by_date = tables.get('agg_by_date')
agg_by_week = tables.get('agg_by_week')
agg_by_month = tables.get('agg_by_month')
agg_by_date_topic = tables.get('agg_by_date_topic')
agg_by_week_topic = tables.get('agg_by_week_topic')
agg_by_month_topic = tables.get('agg_by_month_topic')
raw_data = tables.get('raw_data')

# Ensure all date columns in the DataFrame are of type pd.Timestamp
agg_by_date['date'] = pd.to_datetime(agg_by_date['date'])
agg_by_date_topic['date'] = pd.to_datetime(agg_by_date_topic['date'])
agg_by_week['week'] = pd.to_datetime(agg_by_week['week'])
agg_by_week_topic['week'] = pd.to_datetime(agg_by_week_topic['week'])
agg_by_month['month'] = pd.to_datetime(agg_by_month['month'])
agg_by_month_topic['month'] = pd.to_datetime(agg_by_month_topic['month'])

# === Fix and Convert Date Columns ===
# Ensure the 'date' column in raw_data is correctly formatted
raw_data['date'] = pd.to_datetime(raw_data['date'], errors='coerce')
raw_data = raw_data.dropna(subset=['date'])

# Convert date columns in all dataframes
date_columns = {'date': agg_by_date, 'week': agg_by_week, 'month': agg_by_month}
for col_name, table in date_columns.items():
    if table is not None:
        table[col_name] = pd.to_datetime(table[col_name], errors='coerce')
        table.dropna(subset=[col_name], inplace=True)


# First Filter Section
filter_row = st.columns([1, 1])
with filter_row[0]:
    date_range = st.date_input("Select date range", [raw_data['date'].min(), raw_data['date'].max()], key="date_range_filter")
with filter_row[1]:
    aggregation_level = st.selectbox("Select aggregation level", ['Day', 'Week', 'Month'], key="aggregation_level_filter")


# Set default values for start_date and end_date in case only one date is selected
if len(date_range) == 1:
    date_range = (date_range[0], raw_data['date'].max()) 

start_date = pd.to_datetime(date_range[0])
end_date = pd.to_datetime(date_range[1])




# Determine the appropriate date/aggregation column
date_column = 'date' if aggregation_level == 'Day' else 'week' if aggregation_level == 'Week' else 'month'

# Filter data based on selection
filtered_data = filter_data_by_date(tables[f'agg_by_{date_column}'], date_column, start_date, end_date)
filtered_topic_data = filter_data_by_date(tables[f'agg_by_{date_column}_topic'], date_column, start_date, end_date)

# Visualizations
st.plotly_chart(plot_line_chart(filtered_data, date_column, 'count', f"Number of Reviews Over Time ({aggregation_level})", 'Date', 'Number of Reviews'))
st.plotly_chart(plot_line_chart(filtered_data, date_column, 'average', f"Average Score Over Time ({aggregation_level})", 'Date', 'Average Score'))

# Highest and lowest average scores
high_score_data = prepare_score_data(filtered_topic_data)
top_high_scores = high_score_data.sort_values(by='average_score', ascending=False).head(5)
st.plotly_chart(plot_bar_chart(top_high_scores, 'short_name', 'average_score', "Top Topics with Highest Average Scores", 'Topic', 'Average Score', hover_name='topic_name', y_range=[4, 5]))

top_low_scores = high_score_data.sort_values(by='average_score', ascending=True).head(5)
st.plotly_chart(plot_bar_chart(top_low_scores, 'short_name', 'average_score', "Topics with Lowest Average Scores", 'Topic', 'Average Score', hover_name='topic_name', y_range=[1, 5]))

# Second Filter Section
st.markdown("<h2 style='text-align: center;'>Specific Topic Insi22ghts</h2>", unsafe_allow_html=True)


# Calculate min date
min_date = raw_data['date'].min()

# Calculate max date as the last Sunday before or on the actual max date
max_date = raw_data['date'].max()
last_sunday = max_date - pd.Timedelta(days=max_date.weekday() + 1)

# Set up filters
filter_row = st.columns([1, 1, 1])
with filter_row[0]:
    selected_topic = st.selectbox("Select a topic", raw_data['topic_name'].unique(),index=4, key="topic_filter_specific")

with filter_row[1]:
    date_range_specific = st.date_input("Select date range", [min_date, last_sunday], key="date_range_filter_specific")

with filter_row[2]:
    specific_aggregation_level = st.selectbox("Select aggregation level", ['Day', 'Week', 'Month'],index=1, key="aggregation_level_specific")


# Set default values for start_date and end_date in case only one date is selected
if len(date_range_specific) == 1:
    date_range_specific = (date_range_specific[0], last_sunday) 



date_col = 'date' if specific_aggregation_level == 'Day' else 'week' if specific_aggregation_level == 'Week' else 'month'
agg_table = tables[f'agg_by_{date_col}']
agg_topic_table = tables[f'agg_by_{date_col}_topic']

filtered_data_specific = filter_data_by_date(agg_table, date_col, date_range_specific[0], date_range_specific[1])
filtered_topic_data_specific = filter_data_by_date(agg_topic_table, date_col, date_range_specific[0], date_range_specific[1])
filtered_topic_data_specific = filtered_topic_data_specific[filtered_topic_data_specific['topic_name'] == selected_topic]

# Display charts for topic insights
st.plotly_chart(plot_line_chart(filtered_topic_data_specific, date_col, 'count', f"Number of Reviews Over Time ({specific_aggregation_level})", date_col.capitalize(), 'Number of Reviews'))
st.plotly_chart(plot_line_chart(filtered_topic_data_specific, date_col, 'average', f"Average Score Over Time ({specific_aggregation_level})", date_col.capitalize(), 'Average Score'))

# Display filtered reviews table
st.markdown("<h4 style='text-align: center; color: black;'>Read Filtered Reviews</h4>", unsafe_allow_html=True)


filtered_reviews = raw_data[
    (raw_data['date'] >= pd.to_datetime(date_range_specific[0])) & 
    (raw_data['date'] <= pd.to_datetime(date_range_specific[1])) & 
    (raw_data['topic_name'] == selected_topic)
][['review_timestamp', 'topic_name', 'reviewCreatedVersion','score', 'content']]


filtered_reviews.rename(columns={'review_timestamp': 'Review Timestamp', 'topic_name': 'Representation', 'reviewCreatedVersion': 'App Version', 'score': 'Score Given', 'content': 'Full Review Text'}, inplace = True)

#display reviews table
gb = GridOptionsBuilder.from_dataframe(filtered_reviews)
columns_to_center = ['Review Timestamp', 'App Version', 'Score Given']

for column in columns_to_center:
    gb.configure_column(column, cellStyle={'textAlign': 'center'}, headerClass="center-header")

gridOptions = gb.build()

custom_css = {
    ".center-header": {"textAlign": "center"}
}

AgGrid(filtered_reviews, gridOptions=gridOptions, custom_css=custom_css)