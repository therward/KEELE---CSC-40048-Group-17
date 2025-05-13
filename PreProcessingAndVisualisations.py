import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

out_dir = Path.home() / 'Desktop'
if not out_dir.exists():
    print(f"issue: {out_dir} not found. Saving outputs to current directory.")
    out_dir = Path.cwd()
os.makedirs(out_dir, exist_ok=True)
print(f"Outputs will be written to: {out_dir}\n")

file_path = out_dir / 'Covid_Data.csv'  # (This may need changing if the file moves from desktop)
df = pd.read_csv(file_path)


## The data pre processing steps: 
    
    

# firstly removing duplicated records and print how many it removed 
duplicate_count = df.duplicated().sum()
df = df.drop_duplicates(keep='last')
print(f"Duplicate entries removed: {duplicate_count}")


# look if theres any missing data - if there is then we can update this to act upon it
missing_counts = df.isna().sum()

print("\nMissing value counts:")
print(missing_counts)


#this check ensures the ages of entrys are what we expect 
initial_row_count = len(df)
df = df[df['AGE'].between(0, 120)]
removed = initial_row_count - len(df)
print(f"\nRemoved {removed} rows with ages outside 0–120")



# creating a died column to make it easier to identify if someone died or surivived
# at the moment the only way to tell if someone survived is if theres 9999-99-99 in the date of date died 
df['DIED'] = df['DATE_DIED'].apply(lambda x: 0 if x == '9999-99-99' else 1)



# this is just a check to ensure all records have been recorded properly 
# it will remove any males marked as pregnant as it would indicate incorrect data entry 
pregnant_males = df[(df['SEX'] == 2) & (df['PREGNANT'] == 1)]
print(f"\n inccorect  cases (pregnant males): {len(pregnant_males)}")
df = df[~((df['SEX'] == 2) & (df['PREGNANT'] == 1))]



# saves the cleaned data just to make it clearer that the pre-processing has worked properly 
clean_path = out_dir / 'Covid_Data_Cleaned.csv'
df.to_csv(clean_path, index=False)
print(f"\nCleaned dataset saved to: {clean_path}")



# visualisations setion # 


# GRAPH 1 - covid death rate by age group ###


df = pd.read_csv('Covid_Data_Cleaned.csv')

# putting ages into groups of 10
bins = list(range(0, 101, 10))
labels = [f"{i}-{i+9}" for i in bins[:-1]]
df['AGE_GROUP'] = pd.cut(df['AGE'], bins=bins, labels=labels, right=False)

death_by_age = df.groupby('AGE_GROUP')['DIED'].mean() * 100  # percentage for y axis 

# plotting
plt.figure(figsize=(10, 6))
death_by_age.plot(kind='line', marker='o', color='red')
plt.title('COVID Death Rate by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Death Rate (%)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()



# GRAPH 2 - death rate by gender ###



# mapping the vlaues into genders (1 = female then 2 = male)
sex_map = {1: 'Female', 2: 'Male'}
df['SEX_LABEL'] = df['SEX'].map(sex_map)


death_rate_by_gender = df.groupby('SEX_LABEL')['DIED'].mean() * 100

# plotting
plt.figure(figsize=(8, 6))
death_rate_by_gender.plot(kind='bar', color=['purple', 'teal'], edgecolor='black')

plt.title('COVID Death Rate by Gender')
plt.ylabel('Death Rate (%)')
plt.xlabel('Gender')
plt.ylim(0, death_rate_by_gender.max() + 5)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()



# GRAPH 3 - Intubation rate by age ###


# Filter valid intubation values
df = df[df['INTUBED'].isin([1, 2])]

# Group by AGE and calculate the percentage of patients who were intubated
intubation_by_age = df.groupby('AGE')['INTUBED'].apply(lambda x: (x == 1).mean() * 100)

# Plotting
plt.figure(figsize=(12, 6))
intubation_by_age.plot(kind='line', marker='o', color='darkred')
plt.title('Intubation Rate by Age')
plt.xlabel('Age')
plt.ylabel('Intubation Rate (%)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()







# GRAPH 4 - care type for pregnant patients ###

import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('Covid_Data_Cleaned.csv')

# Filter to pregnant patients only
preg_df = df[df['PREGNANT'] == 1]

# Count how many were sent home vs. hospitalized
care_counts = preg_df['PATIENT_TYPE'].value_counts().sort_index()

# Map labels for readability
care_labels = {1: 'Sent Home', 2: 'Hospitalized'}
care_counts.index = care_counts.index.map(care_labels)

# Plot
plt.figure(figsize=(8, 6))
care_counts.plot(kind='bar', color='mediumseagreen', edgecolor='black')

plt.title('Care Type for Pregnant Patients')
plt.xlabel('Care Type')
plt.ylabel('Number of Patients')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()







# GRAPH 5 - pie chart for patient outcome if they died or survived ###

#Counting deaths and survival
death_counts = df['DIED'].value_counts().sort_index()
labels = {0: 'Survived', 1: 'Died'}
death_counts.index = death_counts.index.map(labels)

# plotting
plt.figure(figsize=(6, 6))
plt.pie(death_counts, labels=death_counts.index, autopct='%1.1f%%',
        startangle=90, colors=['lightgreen', 'crimson'])
plt.title('Patient Outcome: Death vs. Survival')
plt.axis('equal')
plt.tight_layout()
plt.show()




# GRAPH 6 - hospitalisation rate by tabacco use and age groups ####


# the filters 
df = df[df['TOBACCO'].isin([1, 2]) & df['PATIENT_TYPE'].isin([1, 2]) & df['AGE'].between(0, 120)]
df['TOBACCO_LABEL'] = df['TOBACCO'].map({1: 'Smoker', 2: 'Non-Smoker'})
bins = list(range(0, 101, 10))
labels = [f'{i}-{i+9}' for i in bins[:-1]]
df['AGE_GROUP'] = pd.cut(df['AGE'], bins=bins, labels=labels, right=False)

grouped = df.groupby(['AGE_GROUP', 'TOBACCO_LABEL'])['PATIENT_TYPE'].apply(
    lambda x: (x == 2).mean() * 100
).unstack()

#plotting
grouped.plot(kind='bar', figsize=(12, 6), edgecolor='black')
plt.title('Hospitalization Rate by Tobacco Use and Age Group')
plt.xlabel('Age Group')
plt.ylabel('Hospitalization Rate (%)')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.legend(title='Tobacco Use')
plt.tight_layout()
plt.show()





# GRAPH 7 - hospitalisation rate by obesity and age group ####


# filtering
df = df[df['OBESITY'].isin([1, 2]) & df['PATIENT_TYPE'].isin([1, 2]) & df['AGE'].between(0, 120)]
df['HOSPITALIZED'] = df['PATIENT_TYPE'].apply(lambda x: 1 if x == 2 else 0)
df['OBESITY_LABEL'] = df['OBESITY'].map({1: 'Obese', 2: 'Not Obese'})

# age groups
bins = list(range(0, 101, 10))
labels = [f'{i}-{i+9}' for i in bins[:-1]]
df['AGE_GROUP'] = pd.cut(df['AGE'], bins=bins, labels=labels, right=False)
grouped = df.groupby(['AGE_GROUP', 'OBESITY_LABEL'])['HOSPITALIZED'].mean().unstack() * 100

# plotting
grouped.plot(kind='bar', figsize=(12, 6), edgecolor='black', color=['tomato', 'mediumseagreen'])
plt.title('Hospitalization Rate by Obesity and Age Group')
plt.xlabel('Age Group')
plt.ylabel('Hospitalization Rate (%)')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.legend(title='Obesity Status')
plt.tight_layout()
plt.show()



# Graph 8  - pie chart on the death vs survival rate among immunosuppresed patiens ####


# Filter immunosuppressed patients 
immuno_df = df[df['INMSUPR'] == 1]

# counting death and survival 
death_counts = immuno_df['DIED'].value_counts().sort_index()
death_counts.index = death_counts.index.map({0: 'Survived', 1: 'Died'})

# plotting
plt.figure(figsize=(6, 6))
plt.pie(death_counts, labels=death_counts.index, autopct='%1.1f%%', startangle=90, colors=['orange', 'crimson'])
plt.title('Death vs Survival Among Immunosuppressed Patients')
plt.axis('equal')
plt.tight_layout()
plt.show()



# Graph 9 - ICU admission rate by pneumonia patients  ####

# filters 
df = df[df['ICU'].isin([1, 2]) & df['PNEUMONIA'].isin([1, 2])]
df['ICU_ADMITTED'] = df['ICU'].apply(lambda x: 1 if x == 1 else 0) # 1 = admitted to ICU, 0 = not
df['PNEUMONIA_LABEL'] = df['PNEUMONIA'].map({1: 'Had Pneumonia', 2: 'No Pneumonia'})

# calculating the admission rate
icu_rate = df.groupby('PNEUMONIA_LABEL')['ICU_ADMITTED'].mean() * 100

#plotting
plt.figure(figsize=(8, 6))
icu_rate.plot(kind='bar', color=['indianred', 'lightskyblue'], edgecolor='black')
plt.title('ICU Admission Rate by Pneumonia Status')
plt.xlabel('Pneumonia Status')
plt.ylabel('ICU Admission Rate (%)')
plt.ylim(0, icu_rate.max() + 10)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()



# Graph 10 - conditions present in deceased patients  ### 


# Filter for only patients that died 
dead_df = df[df['DIED'] == 1]

# listing conditions 
condition_cols = [
    'OBESITY', 'TOBACCO', 'DIABETES',
    'COPD', 'RENAL_CHRONIC', 'CARDIOVASCULAR',
    'INMSUPR', 'ASTHMA', 'OTHER_DISEASE'
]

condition_counts = (dead_df[condition_cols] == 1).sum().sort_values(ascending=False)

# converting them to lowercase to make it visually nicer 
labels = {
    'OBESITY': 'Obesity',
    'TOBACCO': 'Smoker',
    'DIABETES': 'Diabetes',
    'COPD': 'COPD',
    'RENAL_CHRONIC': 'Renal Disease',
    'CARDIOVASCULAR': 'Cardiovascular',
    'INMSUPR': 'Immunosuppressed',
    'ASTHMA': 'Asthma',
    'OTHER_DISEASE': 'Other Disease'
}
condition_counts.index = condition_counts.index.map(labels)

# plotting
plt.figure(figsize=(10, 6))
condition_counts.plot(kind='bar', color='crimson', edgecolor='black')

plt.title('Conditions Present in Deceased Patients')
plt.ylabel('Number of Patients')
plt.xlabel('Condition')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


