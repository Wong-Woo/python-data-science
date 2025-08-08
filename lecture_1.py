import fire
import pandas as pd
import warnings
import numpy as np

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

# uniformly sampling by RAD(Radial Expressway Accessibility Index) value
def print_uniformly_sampling():import pandas as pd
import numpy as np
import warnings
import fire

warnings.filterwarnings("ignore")

# 전역 데이터 로드
src_path = './kaggle_boston_price.csv'
df = pd.read_csv(src_path, sep=',')

def data_overview():
    """데이터 개요와 결측값 확인"""
    print("=== 데이터 개요 ===")
    print(df.head(5))
    print("\n=== 결측값 확인 ===")
    print(df.isnull().sum())

def simple_sampling(n=10, random_state=47):
    """단순 표본추출"""
    print(f"=== 단순 표본추출 (n={n}) ===")
    sampling_results_df = df.sample(n=n, replace=False, random_state=random_state)
    print(sampling_results_df)
    print(f"샘플 크기: {len(sampling_results_df)}")

def check_groupby_sample():
    """그룹화된 데이터 확인"""
    print("=== 그룹화된 데이터 확인 ===")
    grouped = df.groupby('RAD')
    grouped.apply(lambda x: print(x))
    print("=============================================")
    grouped.apply(lambda x: print(x.sample(2)))


def uniform_sampling(sample_per_group=2):
    """균등표본추출 - RAD 그룹별로 균등하게 추출"""
    print("=== 균등표본추출 ===")
    print(f"RAD 최대값: {df['RAD'].max()}")
    print(f"RAD 최소값: {df['RAD'].min()}")
    
    print("\n각 RAD 그룹별 데이터 건수:")
    print(df['RAD'].value_counts())
    
    print(f"\n각 그룹에서 {sample_per_group}개씩 균등 샘플링:")
    result = df.groupby('RAD', group_keys=False).apply(lambda x: x.sample(sample_per_group))
    print(result)

def stratified_sampling(total_samples=10):
    """층화추출 - 각 그룹의 비율에 따라 추출"""
    print("=== 층화추출 ===")
    
    # 각 그룹별 비율 확인
    total = len(df)
    MAX = df['RAD'].max() + 1
    print("각 RAD 그룹별 비율:")
    for i in range(MAX):
        ratio = len(df.loc[df["RAD"]==i]) / total
        print(f'RAD 지수 : {i}, 비율 {ratio:.4f}')
    
    print(f"\n총 {total_samples}개 샘플을 비율에 따라 층화추출:")
    result = df.groupby('RAD').apply(lambda x: x.sample(int(round(total_samples*len(x)/len(df)))))
    print(result)
    
    print(f"\n인덱스 재설정 후 셔플:")
    shuffled_result = df.groupby('RAD', group_keys=False).apply(
        lambda x: x.sample(int(round(total_samples*len(x)/len(df))))).sample(frac=1).reset_index(drop=True)
    print(shuffled_result)

def conditional_extraction(nox_threshold=0.5, rad_threshold=4):
    """조건을 만족하는 레코드 추출"""
    print("=== 조건부 데이터 추출 ===")
    
    # 단일 조건
    print(f"NOX <= {nox_threshold} 조건:")
    condition_df1 = df.loc[df['NOX'] <= nox_threshold]
    print(condition_df1.head(5))
    print(f"추출된 레코드 수: {len(condition_df1)}")
    
    # 복합 조건
    print(f"\nNOX <= {nox_threshold} AND RAD <= {rad_threshold} 조건:")
    condition_df2 = df.loc[(df['NOX'] <= nox_threshold) & (df['RAD'] <= rad_threshold)]
    print(condition_df2.head(5))
    print(f"추출된 레코드 수: {len(condition_df2)}")

def data_split():
    """데이터 분할 - TAX 중앙값 기준"""
    print("=== 데이터 분할 ===")
    median_tax = df['TAX'].median()
    print(f"TAX 중앙값: {median_tax}")
    
    A_df = df.loc[df['TAX'] <= median_tax]
    print("A_df:")
    print(A_df)
    B_df = df.loc[df['TAX'] > median_tax]
    print("B_df:")
    print(B_df)

    print(f"A_df (TAX <= {median_tax}): {len(A_df)}개")
    print(f"B_df (TAX > {median_tax}): {len(B_df)}개")
    
    return A_df, B_df  # 이 부분이 빠져있었음!

def data_concat():
    """데이터 추가/연결"""
    print("=== 데이터 연결 ===")
    A_df, B_df = data_split()
    
    join_df = pd.concat([A_df, B_df], ignore_index=True)
    print(f"A_df len: {len(A_df)}, B_df len: {len(B_df)}, join_df len: {len(join_df)}")

def data_merge():
    """데이터 병합"""
    print("=== 데이터 병합 ===")
    df1 = df.loc[:3, 'TOWN':'LAT']
    df2 = df.loc[:3, ['LON', 'LAT', 'CMEDV']]
    
    print("df1:")
    print(df1)
    print("\ndf2:")
    print(df2)
    
    merge_df = df1.merge(df2)
    print("\n병합 결과:")
    print(merge_df)

def run_all():
    """모든 분석 실행"""
    print("모든 분석을 순차적으로 실행합니다...\n")
    data_overview()
    print("\n" + "="*50 + "\n")
    simple_sampling()
    print("\n" + "="*50 + "\n")
    uniform_sampling()
    print("\n" + "="*50 + "\n")
    stratified_sampling()
    print("\n" + "="*50 + "\n")
    conditional_extraction()
    print("\n" + "="*50 + "\n")
    data_split()
    print("\n" + "="*50 + "\n")
    data_concat()
    print("\n" + "="*50 + "\n")
    data_merge()

if __name__ == '__main__':
    fire.Fire()

    print(df['RAD'].max())
    print(df['RAD'].min())

    # count by rad level
    print(df['RAD'].value_counts())

    # uniformly sampling 2 data by rad value
    print(df.groupby('RAD', group_keys=False).apply(lambda x: x.sample(2)))

# stratified sampling by RAD value
def print_stratified_sampling():
    # 
    total = len(df)
    MAX = df['RAD'].max() + 1
    for i in range(MAX):
        print(f'RAD 지수 : {i}, 비율 {len(df.loc[df["RAD"]==i])/total}')

    N = 10
    print(df.groupby('RAD').apply(lambda x: x.sample(int(np.rint(N*len(x)/len(df))))))

# Make all functions the runnable with cli
if __name__ == '__main__':
  fire.Fire()