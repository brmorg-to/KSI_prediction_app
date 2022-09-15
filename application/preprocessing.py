import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC

def transform(dataset):
    #%%
    # Load the KSi Dataset
    df_ksi = pd.read_csv(dataset)
    # In[277]:
    # Null values are informed as strings
    df_ksi['OFFSET'][0]
    # In[278]:
    type(df_ksi['OFFSET'][0])
    # In[279]:
    # Set all "<Null>" strings to NaN
    df_ksi = df_ksi.replace('<Null>', np.nan, regex = True)
    # In[282]:
    # Move target class to the last position in the dataframe
    new_cols = ['X', 'Y', 'INDEX_', 'ACCNUM', 'YEAR', 'DATE', 'TIME', 'HOUR', 'STREET1',
        'STREET2', 'OFFSET', 'ROAD_CLASS', 'DISTRICT', 'WARDNUM', 'DIVISION',
        'LATITUDE', 'LONGITUDE', 'LOCCOORD', 'ACCLOC', 'TRAFFCTL', 'VISIBILITY',
        'LIGHT', 'RDSFCOND', 'IMPACTYPE', 'INVTYPE', 'INVAGE',
        'INJURY', 'FATAL_NO', 'INITDIR', 'VEHTYPE', 'MANOEUVER', 'DRIVACT',
        'DRIVCOND', 'PEDTYPE', 'PEDACT', 'PEDCOND', 'CYCLISTYPE', 'CYCACT',
        'CYCCOND', 'PEDESTRIAN', 'CYCLIST', 'AUTOMOBILE', 'MOTORCYCLE', 'TRUCK',
        'TRSN_CITY_VEH', 'EMERG_VEH', 'PASSENGER', 'SPEEDING', 'AG_DRIV',
        'REDLIGHT', 'ALCOHOL', 'DISABILITY', 'POLICE_DIVISION', 'HOOD_ID',
        'NEIGHBOURHOOD', 'ObjectId', 'ACCLASS']
    df_ksi = df_ksi.reindex(columns = new_cols)
    # In[284]:
    # Number of columns in the dataset
    num_columns = len(df_ksi.columns)
    # In[285]:
    # Number of columns that contain at least one missing value
    num_missing_val_columns = len(df_ksi.isna().sum()[df_ksi.isna().sum()>0])
    # ### Columns and their respective numbers of missing values
    # In[286]:
    df_ksi.isna().sum()[df_ksi.isna().sum()>0]
    # In[289]:
    # Proportion of columns with ate least one missing value
    # print(f'{round((num_missing_val_columns / num_columns)*100,2)}% have at least one missing value')
    # In[290]:
    df_ksi.drop(['X', 'Y', 'INDEX_'], axis = 1, inplace = True)
    # `Drop X and Y since they are a different scale for latitude and longitude
    # INDEX_ is also dropped for it's lack of statistical value.`
    # `The DATE column is split into DAY, MONTH, AND YEAR. The latter is dropped`
    # In[292]:
    df_ksi['DATE'] = pd.to_datetime(df_ksi['DATE'], format = '%Y/%m/%d %H:%M:%S')
    # In[293]:
    df_ksi.insert(1, 'MONTH', df_ksi['DATE'].dt.month)
    # In[294]:
    df_ksi.insert(2, 'DAY', df_ksi['DATE'].dt.day)
    # In[295]:
    df_ksi.drop(['YEAR', 'DATE', 'HOUR'], axis = 1, inplace = True)
    # In[297]:
    df_ksi.OFFSET.value_counts()
    # In[298]:
    df_ksi.OFFSET.isna().sum() /len(df_ksi.OFFSET)
    # In[299]:
    # Drop ACCNUM and OFFSET to reduce model complexity. 
    df_ksi.drop(['ACCNUM','OFFSET'], axis = 1, inplace = True)
    # In[300]:
    # Switch all NaN values in Road_Class
    df_ksi.ROAD_CLASS.replace(to_replace = np.nan, value = 'Road Type Unavailable', inplace = True)
    # In[301]:
    df_ksi.ROAD_CLASS.value_counts()
    # In[302]:
    df_ksi.DISTRICT.value_counts()
    # In[303]:
    # Merge 'Toronto East York' with 'Toronto and East York' 
    df_ksi.DISTRICT.replace(to_replace = 'Toronto East York', value = 'Toronto and East York', inplace = True)
    # In[304]:
    # Replace NaN
    df_ksi.DISTRICT.replace(to_replace = np.nan, value = 'District_Not_Informed', inplace = True)
    # In[305]:
    # Drop 'WARDNUM' and 'DIVISION,' since their intrinsic values overlap with
    # other columns that better inidcate location of the accident
    df_ksi.drop(['WARDNUM', 'DIVISION'], axis = 1, inplace = True)
    #Imputer
    imp_freq = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    #Fit_transform Imputer
    df_ksi[['STREET1','STREET2']] =imp_freq.fit_transform(df_ksi[['STREET1','STREET2']])
    # ACCLOC has a much larger number of NaN than LOCCOORD
    # `Whenever LOCCORD has an NaN value and ACCLOC does note, we'll fill
    # the first column with the values from the latter`
    # In[310]:
    #Fill NaN values in LOCCOORD with correspondent values in ACCLOC
    df_ksi['LOCCOORD'].fillna(df_ksi.ACCLOC[df_ksi['LOCCOORD'].isna()], inplace=True)
    # In[313]:
    # Replace NaN values
    df_ksi.LOCCOORD.replace(to_replace = np.nan, value = 'Loccoord_Not_Informed', inplace = True)
    # In[314]:
    df_ksi.drop('ACCLOC', axis = 1, inplace = True)
    # In[317]:
    # Replace NaN values
    df_ksi.TRAFFCTL.replace(to_replace = np.nan, value = 'Traffctl_Not_Informed', inplace = True)
    # In[319]:
    # Merge 'Other and NaN into a new category'
    df_ksi.VISIBILITY.replace(to_replace = ['Other', np.nan], value = 'Other_Visibility', inplace = True)
    # In[324]:
    # Merge 'Other' and NaN into 'Other_Road_Conditions'
    df_ksi.RDSFCOND.replace(to_replace = ['Other', np.nan], value = 'Other_Road_Conditions', inplace = True)
    # In[326]:
    # Merge 'Other' and NaN into 'Other_Impact_Type'
    df_ksi.IMPACTYPE.replace(to_replace = ['Other', np.nan], value = 'Other_Impact_Type', inplace = True)
    # In[329]:
    # Drop Involvement type, since it seems to overlap with other features' information 
    df_ksi.drop('INVTYPE', axis = 1, inplace = True)
    # In[332]:
    # Inkury also seems to be relevant
    # df_ksi.INJURY.replace(to_replace = np.nan, value = 'Injury_Not_Disclosed', inplace = True)
    df_ksi.drop('INJURY', axis = 1, inplace = True)
    # In[334]:
    # Fatal number is merely an identifier
    df_ksi.drop('FATAL_NO', axis = 1, inplace = True)
    # In[337]:
    # Drop 'INITDIR' to reduce dimension and model complexity
    df_ksi.drop('INITDIR', axis = 1, inplace = True)
    # In[339]:
    # Drop 'VEHTYPE' to reduce dimensions and model complexity
    # The most relevant information contained here is replicated in other columns
    df_ksi.drop('VEHTYPE', axis = 1, inplace = True)
    # In[342]:
    # Drop 'MANOUVER' to reduce dimensions and model complexity
    df_ksi.drop('MANOEUVER', axis = 1, inplace = True)
    # In[343]:
    # Driver Action seems relevant
    df_ksi.DRIVACT.replace(to_replace = ['Other', np.nan], value = 'Other_Driver_Action', inplace = True)
    # In[346]:
    df_ksi.DRIVCOND.replace(to_replace = ['Unknown', 'Other', np.nan], value = 'Driver_Condition_Unkown', inplace = True)
    # In[348]:
    # Drop 'PEDTYPE' to reduce dimensions and model complexity
    df_ksi.drop('PEDTYPE', axis = 1, inplace = True)
    # In[349]:
    # Drop 'PEDACT' to reduce dimensions and model complexity
    df_ksi.drop('PEDACT', axis = 1, inplace = True)
    # In[350]:
    # Drop 'PEDCOND' to reduce dimensions and model complexity
    df_ksi.drop('PEDCOND', axis = 1, inplace = True)
    # In[351]:
    # Drop 'CYCLISTYPE' to reduce dimensions and model complexity
    df_ksi.drop('CYCLISTYPE', axis = 1, inplace = True)
    # In[353]:
    # Drop 'CYCACT' to reduce dimensions and model complexity
    df_ksi.drop('CYCACT', axis = 1, inplace = True)
    # In[355]:
    # Drop 'CYCCOND' due to the number of NaN values and to reduce model complexity
    df_ksi.drop('CYCCOND', axis = 1, inplace = True)
    # ### Keep the categories of those involved in the accident
    # In[99]:
    df_ksi.PEDESTRIAN = df_ksi.PEDESTRIAN.map({'Yes': 1,
                                            np.nan: 0})
    # In[101]:
    df_ksi.CYCLIST = df_ksi.CYCLIST.map({'Yes': 1,
                                            np.nan: 0})
    # In[104]:
    df_ksi.AUTOMOBILE = df_ksi.AUTOMOBILE.map({'Yes': 1,
                                            np.nan: 0})
    # In[106]:
    df_ksi.MOTORCYCLE = df_ksi.MOTORCYCLE.map({'Yes': 1,
                                            np.nan: 0})
    # In[109]:
    df_ksi.TRUCK = df_ksi.TRUCK.map({'Yes': 1,
                                    np.nan: 0})
    # In[113]:
    df_ksi.TRSN_CITY_VEH = df_ksi.TRSN_CITY_VEH.map({'Yes': 1,
                                                    np.nan: 0})
    # In[116]:
    df_ksi.EMERG_VEH = df_ksi.EMERG_VEH.map({'Yes': 1,
                                            np.nan: 0})
    # In[119]:
    df_ksi.PASSENGER = df_ksi.PASSENGER.map({'Yes': 1,
                                            np.nan: 0})
    # ### Speeding, Agressive, red light, and alcohol driving seem empirically important for our model
    # In[122]:
    df_ksi.SPEEDING = df_ksi.SPEEDING.map({'Yes': 1,
                                            np.nan: 0})
    # In[124]:
    df_ksi.AG_DRIV = df_ksi.AG_DRIV.map({'Yes': 1,
                                        np.nan: 0})
    # In[127]:
    df_ksi.REDLIGHT = df_ksi.REDLIGHT.map({'Yes': 1,
                                            np.nan: 0})
    # In[129]:
    df_ksi.ALCOHOL = df_ksi.ALCOHOL.map({'Yes': 1,
                                        np.nan: 0})
    # In[131]:
    df_ksi.DISABILITY = df_ksi.DISABILITY.map({'Yes': 1,
                                            np.nan: 0})
    # In[356]:
    # Drop these location columns and IDs to simplify the model
    df_ksi.drop(['HOOD_ID','ObjectId', 'POLICE_DIVISION', 'NEIGHBOURHOOD'], axis = 1, inplace = True)
    # In[358]:
    # Create a copy of the dataset to prepare for transformation
    df_pipeline = df_ksi.copy()
    # In[360]:
    # Categorical Features
    df_categorical = df_pipeline.select_dtypes(include = ['object']).drop('ACCLASS', axis = 1)
    # In[361]:
    # Numeric Features
    df_numeric = df_pipeline[['MONTH', 'DAY', 'TIME', 'LATITUDE', 'LONGITUDE']]
    # In[362]:
    # Reduce the target variable to two classes
    df_pipeline.ACCLASS.replace(to_replace = ['Property Damage Only', 'Non-Fatal Injury'], value = 'Non-Fatal', inplace = True)
    # In[363]:
    df_pipeline.ACCLASS.value_counts()
    # In[364]:
    # Convert the dependent variable to numeric
    classification = pd.get_dummies(df_pipeline['ACCLASS'])
    df_pipeline = pd.concat([df_pipeline, classification], axis = 1)
    df_pipeline.drop('ACCLASS', axis = 1, inplace = True)
    # In[365]:
    df_pipeline.drop('Non-Fatal', axis = 1, inplace = True)
    # In[367]:
    # Instantiate the encoder
    # encoder = OneHotEncoder(drop = 'first', handle_unknown='ignore')
    # In[368]:
    # ColumnTransformer
    num_attributes = df_numeric.columns
    cat_attributes = df_categorical.columns
    transformer = ColumnTransformer([
        ('encoder', OneHotEncoder(drop = 'first', handle_unknown='ignore'), cat_attributes),
        ('standardizer', StandardScaler(), num_attributes)],
        remainder='passthrough',
        verbose_feature_names_out=False)
    # In[369]:
    transformer.transformers
    # In[370]:
    # Features and Target
    features = df_pipeline.drop('Fatal', axis = 1)
    target = df_pipeline.Fatal
    # In[371]:
    # Split into Training and Test
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.20, random_state = 98)

    X_test.to_csv('application/dataset/X_test_2.csv')
    y_test.to_csv('application/dataset/y_test_2.csv')

    return X_test, y_test
