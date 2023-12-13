import pandas as pd

def convert_last_column_to_categorical(file_path, new_file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Get the unique values in the last column
    unique_values = df.iloc[:, -1].unique()
    
    # Create a mapping dictionary for categorical numbering
    mapping = {value: i+1 for i, value in enumerate(unique_values)}
    
    # Convert the last column to categorical numbering
    df.iloc[:, -1] = df.iloc[:, -1].map(mapping)
    
    # Save the modified DataFrame to a new CSV file
    df.to_csv(new_file_path, index=False)

# Example usage
convert_last_column_to_categorical('final_datasets/breast_cancer_binned.csv', 'final_datasets/breast_cancer_categorical.csv')
