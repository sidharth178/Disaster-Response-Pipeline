# import necessary libraries
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


# load data 
def load_data(messages_filepath, categories_filepath):
  
  # load two csv files
  messages = pd.read_csv(messages_filepath)
  categories = pd.read_csv(categories_filepath)
  return pd.merge(messages, categories, on='id', how='inner')


# data pre-processing
def clean_data(df):

  categories = df["categories"].str.split(';', expand=True)
  # select the first row of the categories dataframe
  row = categories[0:1]
  # use this row to extract a list of new column names for categories.
  category_col_names = row.apply(lambda x: x.str[:-2]).values.tolist()
  # rename the columns of `categories` dataframe
  categories.columns = category_col_names

  # Convert category values to just numbers 0 or 1.
  for column in categories:
      # set each value to be the last character of the string
      categories[column] = categories[column].str[-1]
      # change column data type from string to numeric
      categories[column] = categories[column].astype(int)

  # drop the original categories column from `df`
  df.drop('categories', axis=1, inplace = True)
  # concatenate the original dataframe with the new `categories` dataframe
  df = pd.concat([df, categories], axis=1)
  # drop duplicates
  df = df.drop_duplicates()
  return df


# save cleaned data  
def save_data(df, database_filename):
  
  # save a dataframe to sqlite database
  print('Save {} to {} database: '.format(df, database_filename))
  engine = create_engine('sqlite:///{}'.format(database_filename))
  df.to_sql('DisasterResponse', engine,if_exists = 'replace', index=False)


def main():
  
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()