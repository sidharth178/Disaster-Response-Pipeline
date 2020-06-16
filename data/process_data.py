# import all libraries here
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    ''' This is load data function which is used to load the data from files .
  
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, left_on='id', right_on='id', how='outer')
    return df

def clean_data(df):
    ''' This is clean data function which helps in cleaning the data.'''
    categories = df.categories.str.split(';', expand = True)
    row = categories.loc[0]
    category_colnames = row.apply(lambda x: x[:-2]).values.tolist()
    categories.columns = category_colnames
    categories.related.loc[categories.related == 'related-2'] = 'related-1'
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1]
        categories[column] = pd.to_numeric(categories[column])
    df.drop('categories', axis = 1, inplace = True)
    df = pd.concat([df, categories], axis = 1)
    df.drop_duplicates(subset = 'id', inplace = True)
    return df

def save_data(df, database_filepath):
    ''' This is save data function where we link the datafram to  the sqlite and save it in the database.'''
    engine = create_engine('sqlite:///' + database_filepath)
    df.to_sql('FigureEight', engine, index=False)

def main():
    ''' This is main data processing function which grab the paths of files .
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)


        df = clean_data(df)


        save_data(df, database_filepath)

        print('database saved!')

    else:
        print('Pmissing paths!')


if __name__ == '__main__':
    main()
