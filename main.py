import fire
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

src_path = './kaggle_boston_price.csv'
df = pd.read_csv(src_path, sep=',')

# print head of table
def print_head(n):
    print(df.head(n))

# print the count of missing value
def print_null():
    print(df.isnull().sum())

# simple random sampling
def print_simple_random_sampling():
    sampling_results_df = df.sample(n=10, replace=False, random_state=47)
    # replace : 복원추출여부 / random_state : 표집 시드(표본 id)
    print(sampling_results_df)
    print(len(sampling_results_df))

# systematic sampling
# systematically sampling with RAD(Radial Expressway Accessibility Index)
def print_systematic_sampling():
    print(df['RAD'].max())
    print(df['RAD'].min())
    print(df['RAD'].value_counts())

    # 
    print(df.groupby('RAD', group_keys=False).apply(lambda x: x.sample(2)))

# Make all functions the runnable with cli
if __name__ == '__main__':
  fire.Fire()