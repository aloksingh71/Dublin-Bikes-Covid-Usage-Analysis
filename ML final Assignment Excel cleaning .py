#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import pandas as pd
import re

#reading files where  all the dublin dataset is present  
directory = r'ML final assignment/before cleaning'


cleaned_directory = r'ML final assignment/after cleaning'


if not os.path.exists(cleaned_directory):
    os.makedirs(cleaned_directory)

#finding year from name of file  and cleaning data  removing empty cells 
for filename in os.listdir(directory):
    if filename.endswith(".csv") and (filename.startswith("dublinbikes_") or filename.startswith("dublinbike-historical-data-")):
        print(f"Processing file: {filename}")
        file_path = os.path.join(directory, filename)
        

        year_match_1 = re.search(r'dublinbike-historical-data-(\d{4})-(\d{2})\.csv$', filename)  
        year_match_2 = re.search(r'dublinbikes_(\d{8})_(\d{8})\.csv$', filename)  
        
        if year_match_1:
            year = year_match_1.group(1)
            print(f"Year extracted using pattern 1: {year}")
        elif year_match_2:
            year = year_match_2.group(1)[:4] 
            print(f"Year extracted using pattern 2: {year}")
        else:
            print(f"No year found in filename: {filename}")
            continue  
        
        year_directory = os.path.join(cleaned_directory, year)
        
      
        if not os.path.exists(year_directory):
            os.makedirs(year_directory)
            print(f"Created directory for year {year}")
        
        cleaned_file_path = os.path.join(year_directory, f"cleaned_{filename}")


        df = pd.read_csv(file_path)

 
        print(f"Columns in '{filename}': {df.columns.tolist()}")

# filling empty  cells with NA
        columns_to_fill_na = df.columns.tolist()  
        
        num_na_before_cleaning = df[columns_to_fill_na].isna().sum().sum()  

        df[columns_to_fill_na] = df[columns_to_fill_na].fillna('NA')

       
        df['TIME'] = pd.to_datetime(df['TIME'])
        df['LAST UPDATED'] = pd.to_datetime(df['LAST UPDATED'])


        num_na_after_cleaning = df[columns_to_fill_na].isna().sum().sum()
 #saving file to  cleaned directory 
  
        df.to_csv(cleaned_file_path, index=False)

        print(f"File '{filename}' cleaned and saved to '{cleaned_file_path}'")
        print(f"NA values added in '{filename}': {num_na_after_cleaning - num_na_before_cleaning}")


# In[8]:


import os
import pandas as pd

# data to segeregate data on basis of pre covid during covid and after covid
cleaned_directory = r'ML final assignment/after cleaning'
saving_directory= r'ML final assignment/concatenated data'
if not os.path.exists(saving_directory):
    os.makedirs(saving_directory)

def load_cleaned_data(folder_path):
    data_frames = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv") and filename.startswith("cleaned_"):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)

            df.rename(columns={
                'BIKE_STANDS': 'BIKE STANDS',
                'AVAILABLE_BIKE_STANDS': 'AVAILABLE BIKE STANDS',
                'AVAILABLE_BIKES': 'AVAILABLE BIKES'
            }, inplace=True)

            data_frames.append(df)
    return pd.concat(data_frames, ignore_index=True)


all_data = load_cleaned_data(cleaned_directory)


all_data['TIME'] = pd.to_datetime(all_data['TIME'])


before_covid_start = pd.Timestamp('2018-01-01')
before_covid_end = pd.Timestamp('2020-02-29')
during_covid_start = pd.Timestamp('2020-03-01')
during_covid_end = pd.Timestamp('2021-07-31')
after_covid_start = pd.Timestamp('2021-08-01')
after_covid_end = pd.Timestamp('2023-12-31')


before_covid_data = all_data[(all_data['TIME'] >= before_covid_start) & (all_data['TIME'] <= before_covid_end)]
during_covid_data = all_data[(all_data['TIME'] >= during_covid_start) & (all_data['TIME'] <= during_covid_end)]
after_covid_data = all_data[(all_data['TIME'] >= after_covid_start) & (all_data['TIME'] <= after_covid_end)]


before_covid_data = before_covid_data.drop_duplicates()
during_covid_data = during_covid_data.drop_duplicates()
after_covid_data = after_covid_data.drop_duplicates()
# saving files to directory mentioned above

before_covid_data.to_csv(os.path.join(saving_directory, 'before_covid_data.csv'), index=False)
during_covid_data.to_csv(os.path.join(saving_directory, 'during_covid_data.csv'), index=False)
after_covid_data.to_csv(os.path.join(saving_directory, 'after_covid_data.csv'), index=False)



# In[9]:



#code to remove duplicates from file 
directory = r'ML final assignment/concatenated data'


output_directory = r'ML final assignment/removed duplicates'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)


def load_cleaned_data(folder_path):
    data_frames = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)


            df.rename(columns={
                'BIKE STANDS': 'BIKE_STANDS',
                'AVAILABLE BIKE STANDS': 'AVAILABLE_BIKE_STANDS',
                'AVAILABLE BIKES': 'AVAILABLE_BIKES'
            }, inplace=True)

            data_frames.append(df)
    return pd.concat(data_frames, ignore_index=True)


all_data = load_cleaned_data(directory)
#making key to get compare and check for duplicates 

key_columns = ['STATION ID', 'TIME', 'LAST UPDATED', 'NAME', 'BIKE_STANDS', 'AVAILABLE_BIKE_STANDS',
               'AVAILABLE_BIKES', 'STATUS', 'ADDRESS', 'LATITUDE', 'LONGITUDE']


all_data['Composite_Key'] = all_data[key_columns].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

#Dropping duplicates 
initial_rows = all_data.shape[0]
cleaned_data_without_duplicates = all_data.drop_duplicates(subset='Composite_Key')
rows_removed = initial_rows - cleaned_data_without_duplicates.shape[0]
print(f"{rows_removed} rows removed due to duplicates.")


cleaned_data_without_duplicates = cleaned_data_without_duplicates.drop(columns='Composite_Key')

#saving to  output directory
output_file = os.path.join(output_directory, 'cleaned_concatenated_data_without_duplicates.csv')
cleaned_data_without_duplicates.to_csv(output_file, index=False)

print(f"File saved: {output_file}")

