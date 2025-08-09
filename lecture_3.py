import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import fire

warnings.filterwarnings("ignore")

# 전역 데이터 로드
src_path = './kaggle_boston_price.csv'
df = pd.read_csv(src_path, sep=',')

def data_overview():
    """데이터 개요 및 기본 정보 출력"""
    print("=== 데이터 개요 ===")
    print(df.head(3))
    
    print("\n=== RAD, TAX 컬럼 샘플 ===")
    sample_df = df[['RAD', 'TAX']]
    print(sample_df.head(3))

def grouped_statistics(group_col='RAD', value_col='TAX', figsize=(10, 8)):
    """그룹별 통계량 계산 및 시각화"""
    sample_df = df[[group_col, value_col]]
    
    print(f"=== {group_col}별 {value_col} 평균 ===")
    mean_stats = sample_df.groupby(group_col).mean()
    print(mean_stats)
    
    # 평균 막대그래프
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()
    sns.barplot(x=group_col, y=value_col, data=sample_df, ax=ax)
    plt.title(f'{group_col}별 {value_col} 평균')
    plt.show()
    
    print(f"\n=== {group_col}별 {value_col} 분산 ===")
    var_stats = sample_df.groupby(group_col).var()
    print(var_stats)
    
    # 분산 상자그림
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()
    sns.boxplot(x=group_col, y=value_col, data=sample_df, ax=ax)
    plt.title(f'{group_col}별 {value_col} 분포')
    plt.show()
    
    print(f"\n=== {group_col}별 {value_col} 표준편차 ===")
    std_stats = sample_df.groupby(group_col).std()
    print(std_stats)

def distribution_analysis(column='RM', bins=10, figsize=(8, 8)):
    """분포 분석: 평균, 왜도, 첨도 계산 및 히스토그램"""
    print(f"=== {column} 분포 분석 ===")
    
    print(f"평균 (mean): {df[column].mean()}")
    print(f"왜도 (skew): {df[column].skew()}")
    print(f"첨도 (kurt): {df[column].kurt()}")
    
    # 기본 히스토그램
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()
    df[column].hist(bins=bins, ax=ax)
    plt.title(f'{column} 히스토그램')
    plt.xlabel(column)
    plt.ylabel('빈도')
    plt.show()
    
    # KDE가 포함된 히스토그램
    fig = plt.figure(figsize=(6, 6))
    ax = fig.gca()
    sns.histplot(x=column, kde=True, data=df, bins=bins, ax=ax)
    plt.title(f'{column} 히스토그램 (KDE 포함)')
    plt.show()

def descriptive_statistics(column='TAX'):
    """기술통계량, 최빈값, 사분위수 계산"""
    print(f"=== {column} 기술통계량 ===")
    print(df[column].describe())
    
    print(f"\n=== {column} 최빈값 ===")
    value_counts = df[column].value_counts()
    print("값별 빈도:")
    print(value_counts)
    print(f"최빈값: {value_counts.idxmax()}")
    
    print(f"\n=== {column} 사분위수 ===")
    print(f"1사분위수 (25%): {np.percentile(df[column], 25)}")
    print(f"2사분위수 (50%, 중앙값): {np.percentile(df[column], 50)}")
    print(f"3사분위수 (75%): {np.percentile(df[column], 75)}")

def run_all_analysis():
    """모든 분석 실행"""
    print("모든 통계 분석을 순차적으로 실행합니다...\n")
    
    data_overview()
    print("\n" + "="*50 + "\n")
    
    grouped_statistics()
    print("\n" + "="*30 + "\n")
    
    distribution_analysis()
    print("\n" + "="*30 + "\n")
    
    descriptive_statistics()

if __name__ == '__main__':
    fire.Fire()


