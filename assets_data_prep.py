import joblib
train_medians2 = joblib.load('train_medians.pkl')
print(train_medians2.keys())
import pandas as pd
import numpy as np
import sklearn


#Fixing different formats of property values and converting them to one format
def property_type_fix(df,file_type):
    df = df.copy()

    mask = df['property_type'].str.contains('גג', na=False)
    df.loc[mask, 'property_type'] = 'גג/פנטהאוז'

    mask2 = df['property_type'].str.contains('דירה', na=False)
    df.loc[mask2, 'property_type'] = 'דירה'

    mask3 = df['property_type'].str.contains('גן', na=False)
    df.loc[mask3, 'property_type'] = 'דירת גן'

    mask4 = df['property_type'].str.contains('מרתף', na=False)
    df.loc[mask4, 'property_type'] = 'מרתף/פרטר'

    mask5 = df['property_type'].str.contains('דופלקס', na=False)
    df.loc[mask5, 'property_type'] = 'דופלקס'

    mask6 = df['property_type'].str.contains('סטודיו', na=False)
    df.loc[mask6, 'property_type'] = 'סטודיו/לופט'

    mask7 = df['property_type'].str.contains('פרטי', na=False)
    df.loc[mask7, 'property_type'] = 'בית פרטי/קוטג'

    mask8 = df['property_type'].str.contains('דיור', na=False)
    df.loc[mask8, 'property_type'] = 'יחידת דיור'

    mask9 = df['property_type'].str.contains('משפחתי', na=False)
    df.loc[mask9, 'property_type'] = 'דו משפחתי'

    allowed_values = ['דירה', 'גג/פנטהאוז', 'דירת גן', 'מרתף/פרטר', 'דופלקס',
                      'סטודיו/לופט', 'בית פרטי/קוטג', 'יחידת דיור', 'דו משפחתי']
    if file_type == 'train':
        df = df[df['property_type'].isin(allowed_values)]
        return df
    #we define not to filters the df to the "allowed_values" in the test file because it can cause problems to run the test code.
    #(The function would remove the rows that did not match the "allowed_values" and we would get 2 tables (prediction and test) 
    #with an inconsistent number of rows and it would not be possible to calculate RMSE accuracy.
    if file_type == 'test':
        return df

#fix the NaN values with the right : area* mean_rooms per squar in order to be match the real world. 
def room_num_fix(df):
    df = df.copy()
    df = df[df['area'] > 0]
    mean_rooms_per_sqm = (df['room_num'] / df['area']).mean()
    df.loc[df['room_num'] == 0, 'room_num'] = df.loc[df['room_num'] == 0, 'area'] * mean_rooms_per_sqm
    return df

#We noticed that there were different formats for the floors, so we corrected the format and, using the floor column, completed total_floor.
#example of A BAD FORMAT : 5 from 15 Turn into floor:5 total_floors:15
def quick_floor_fix(df):
    def fix_floor_pair(row):
        value = row['floor']
        current_total = row.get('total_floors', np.nan)
        if pd.isna(value):
            return pd.Series([np.nan, current_total])
        if isinstance(value, (int, float)):
            return pd.Series([int(value), current_total])
        value_str = str(value).lower()
        if "קרקע" in value_str:
            return pd.Series([0, current_total])
        numbers = re.findall(r'\d+', value_str)
        if len(numbers) == 2:
            floor_val, total_val = int(numbers[0]), int(numbers[1])
            total_final = total_val if pd.isna(current_total) else current_total
            return pd.Series([floor_val, total_final])
        elif len(numbers) == 1:
            return pd.Series([int(numbers[0]), current_total])
        else:
            return pd.Series([np.nan, current_total])
    df[['floor', 'total_floors']] = df.apply(fix_floor_pair, axis=1)
    return df

#Some values are 3 digit number and in the digit the total_floor is hiding.
#bad format: 149 . if total_floor is 14 assing the floor to 9 
def correct_floor_encoding_verbose(row):
    floor_val = row['floor']
    total_val = row['total_floors']
    if pd.notna(floor_val) and pd.notna(total_val):
        floor_str = str(int(floor_val))
        total_str = str(int(total_val))
        if floor_str.startswith(total_str) and len(floor_str) > len(total_str):
            remaining = floor_str[len(total_str):]
            if remaining.isdigit():
                return pd.Series([int(remaining), "prefix_removed"])
        if floor_str.endswith(total_str) and len(floor_str) > len(total_str):
            remaining = floor_str[:len(floor_str)-len(total_str)]
            if remaining.isdigit():
                return pd.Series([int(remaining), "suffix_removed"])
    return pd.Series([np.nan, None])

#we assumed by the amount of digits and our knowleg with Tel Aviv that some of the distance values are in meter format so We did a unit conversion.
#from meter to km 
def normalize_distance(val):
    try:
        float_val = float(val)
        if float_val >= 100:
            return float_val / 1000
        else:
            return float_val
    except:
        return np.nan

#searching for duplicate rows and remove them.
def find_duplicate_rows_by_address_and_neighborhood(df):
    return df.drop_duplicates(subset=["address", "neighborhood"])

#fixing NaN values in the monthly_arnona column with the median payment of every neighborhood from a way of thinking of each neighborhood have
#probably the same amount of apartment area so they probably pays the same ampunt of monthly_arnona.
def fill_monthly_arnona_by_neighberhood_median(df):
    # calculating median for each neighborhood 
    medians = df.groupby('neighborhood')['monthly_arnona'].median()
    
    #filing the missing values with the median value
    def fill_row(row):
        if pd.isna(row['monthly_arnona']):
            return medians.get(row['neighborhood'], row['monthly_arnona'])
        return row['monthly_arnona']
    
    df['monthly_arnona'] = df.apply(fill_row, axis=1)
    
    return df

################################################################################################prepare_data function#######################################################################################

train_medians = {}
def prepare_data(df, file_type):
    global train_medians2
    if file_type not in ['train', 'test','pred']:
        raise ValueError("file_type must be 'train' , 'test' or 'pred'")

    #features for the train file 
    features_to_include = ['property_type','neighborhood', 'floor', 'area', 'has_parking', 'has_storage', 'elevator',
       'ac', 'handicap', 'has_safe_room','total_floors', 'building_tax','has_balcony', 'is_renovated',
       'is_furnished','rooms_per_sqm'  ,'distance_group', 'monthly_arnona', 'garden_area','price']
     #features for the test file
    features_to_include2 = ['property_type','neighborhood', 'floor', 'area', 'has_parking', 'has_storage', 'elevator',
       'ac', 'handicap', 'has_safe_room','total_floors', 'monthly_arnona', 'building_tax','has_balcony', 'is_renovated',
       'is_furnished','rooms_per_sqm','distance_group', 'garden_area'] #, 'monthly_arnona'
    
    if file_type == 'train': 
        df = property_type_fix(df,file_type)
        df = room_num_fix(df)
        df = quick_floor_fix(df)
        df[['floor_corrected', 'floor_change_type']] = df.apply(correct_floor_encoding_verbose, axis=1)
        df['floor'] = df['floor_corrected'].combine_first(df['floor'])
        #remove rows with nan after the filling with the correct floor value 
        df = df.dropna(subset=['floor'])
        df = find_duplicate_rows_by_address_and_neighborhood(df)
        
        #remove rows with nan after the filling with the correct total_floor value 
        df = df.dropna(subset=['total_floors'])
        df['garden_area'] = pd.to_numeric(df['garden_area'], errors='coerce').fillna(0)
        df['rooms_per_sqm'] = df['room_num'] / df['area']
        df['distance_from_center_normalized'] = df['distance_from_center'].apply(normalize_distance)
        
        #Handling outliers
        df = df[df['distance_from_center_normalized'] <= 20]
        df = df[(df['area'] >= 18) & (df['area'] <= 300)]
        df = df[(df['price'] >= 700) & (df['price'] <= 50000)]
        
        #filling building_tax nans values with 0 from the assuming that if they were not filled out in the apartment ad,
        #it means that there is no tax payment on the building.
        df['building_tax'] = pd.to_numeric(df['building_tax'], errors='coerce').fillna(0)
        
        #Handling outliers
        df = df[df['building_tax'] <= 3000]
        
        #fixing monthly_arnona column
        df = fill_monthly_arnona_by_neighberhood_median(df)
        
        #Handling outliers- from our knowledge arnona in Tel Aviv is not Greater than 30000
        df = df[df['monthly_arnona'] <= 3000]
        df = df.dropna(subset=['monthly_arnona'])
        
       #Handling outliers
        df = df[df['garden_area'] <= 100]
        
        #Handling outliers- it is unlikly to have a Residential building with more than 30 floors
        df = df[df['total_floors'] <= 30]
        df = df.dropna(subset=['distance_from_center', 'monthly_arnona', 'price'])
        
        #creating a new scaling for the distance_from_center in order to define a common format to effect the price. 
        df['distance_group'] = pd.cut(df['distance_from_center_normalized'],
                                      bins=[0.0, 3.0, 6.0, 10.0, 20.0],
                                      labels=['מאוד קרוב', 'קרוב', 'בינוני', 'רחוק'],
                                      include_lowest=True)
        mapping = {'רחוק': 4, 'בינוני': 3, 'קרוב': 2, 'מאוד קרוב': 1}
        df['distance_group'] = df['distance_group'].map(mapping)
        df = df.dropna(subset=['distance_group'])
        df['distance_group'] = df['distance_group'].astype(int)
        
        # Save the median values from the train to fill them in the test file withount removing rows
        train_medians['monthly_arnona'] = df['monthly_arnona'].median() #לשים קודם לפי שכונה ואז אם לא יהיה אז לפי חציון כללי 
        train_medians['building_tax'] = df['building_tax'].median()
        train_medians['total_floors'] = df['total_floors'].median()
        train_medians['distance_from_center_normalized'] = df['distance_from_center_normalized'].median()
        train_medians['distance_group'] = df['distance_group'].mode()[0]
        return df[features_to_include]
    
    if file_type == 'test':
        
        #Fix wrong written template of propery_type
        df = property_type_fix(df,file_type)
        
        #Fix room number
        df = room_num_fix(df)
        
        #Fix floors format (5 from 15 to floor:5, total_floor:15)
        df = quick_floor_fix(df)
        df[['floor_corrected', 'floor_change_type']] = df.apply(correct_floor_encoding_verbose, axis=1)
        df['floor'] = df['floor_corrected'].combine_first(df['floor'])
        df['floor'] = pd.to_numeric(df['floor'], errors='coerce').fillna(0)
        df['total_floors'] = pd.to_numeric(df['total_floors'], errors='coerce').fillna(train_medians['total_floors'])
        
        #Fill garden area  Nans with 0.  
        df['garden_area'] = pd.to_numeric(df['garden_area'], errors='coerce').fillna(0)
        
        #Creating new feature of room per squar meter
        df['rooms_per_sqm'] = df['room_num'] / df['area']

        #transfer meter to km
        df['distance_from_center_normalized'] = df['distance_from_center'].apply(normalize_distance)
        df['distance_from_center_normalized'] = pd.to_numeric(df['distance_from_center_normalized'], errors='coerce').fillna(train_medians['distance_from_center_normalized'])
         
        df['building_tax'] = pd.to_numeric(df['building_tax'], errors='coerce').fillna(0)
        
        #Fill montly arnona with the median value of each neighberhood
        df = fill_monthly_arnona_by_neighberhood_median(df)
       
        #Creating new feature based on distance in order to make the same pattern to distance.
        df['distance_group'] = pd.cut(df['distance_from_center_normalized'],
                                      bins=[0.0, 3.0, 6.0, 10.0, 20.0],
                                      labels=['מאוד קרוב', 'קרוב', 'בינוני', 'רחוק'],
                                      include_lowest=True)
        mapping = {'רחוק': 4, 'בינוני': 3, 'קרוב': 2, 'מאוד קרוב': 1}
        df['distance_group'] = df['distance_group'].map(mapping)
        df['distance_group'] = df['distance_group'].astype(int)
        
        #Fill Nans values of monthly arnona with its median value
        df['monthly_arnona'] = pd.to_numeric(df['monthly_arnona'], errors='coerce').fillna(train_medians['monthly_arnona'])
         
        
    if file_type == 'pred':
                
        if 'monthly_arnona' not in df.columns:
            df['monthly_arnona'] = train_medians2['monthly_arnona']
        if 'building_tax' not in df.columns:
            df['building_tax'] = 0
        
        # Derived feature
        if 'rooms_per_sqm' not in df.columns:
            df['rooms_per_sqm'] = df['room_num'] / df['area']  
        
        return df[features_to_include2]

