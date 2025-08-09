import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import fire

warnings.filterwarnings("ignore")

# 전역 데이터 로드 및 전처리
src_path = './kaggle_boston_price.csv'
df = pd.read_csv(src_path, sep=',') 

# 데이터 전처리 (전역으로 실행)
def _prepare_data():
    """데이터 전처리 - 내부 함수"""
    global df
    
    # TAX_GRADE 생성
    MAX_TAX = df['TAX'].max()
    MIN_TAX = df['TAX'].min()
    TAX_RANGE = MAX_TAX - MIN_TAX
    Interval = TAX_RANGE / 3
    
    df['TAX_GRADE'] = np.where(
        df['TAX'] >= (MAX_TAX - Interval), 'B', 
        np.where(df['TAX'] > (MIN_TAX + Interval), "M", "L")
    )
    
    # 방 1개당 가격 계산
    df['RM_1'] = df['CMEDV'] / df['RM']

# 데이터 전처리 실행
_prepare_data()

def data_overview():
    """데이터 개요 및 새로운 변수 확인"""
    print("=== 원본 데이터 ===")
    print(df.head(5))
    
    print("\n=== TAX_GRADE 분포 ===")
    print(df['TAX_GRADE'].value_counts())
    
    print("\n=== 방 1개당 가격 (RM_1) ===")
    print(df[['CMEDV', 'RM', 'RM_1']].head(5))

def histogram(column='CRIM', bins=10, color='red', figsize=(10, 6)):
    """히스토그램 생성: 수치형 변수의 분포 확인"""
    print(f"=== {column} 히스토그램 ===")
    
    plt.figure(figsize=figsize)
    plt.hist(df[column], alpha=0.3, bins=bins, rwidth=1, color=color, label=column)
    plt.legend()
    plt.grid()
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.title(f'{column} 히스토그램')
    plt.show()

def scatter_plot(x='RM', y='CMEDV', figsize=(10, 6)):
    """산점도 생성: 두 변수 간의 관계 확인"""
    print(f"=== {x} vs {y} 산점도 ===")
    
    plt.figure(figsize=figsize)
    df.plot.scatter(x=x, y=y, title=f'{x} vs {y} Scatter Plot')
    plt.show()

def pie_chart(column='TAX_GRADE', figsize=(10, 8)):
    """파이차트 생성: 범주형 변수의 분포 확인"""
    print(f"=== {column} 파이차트 ===")
    
    print(f"{column} 분포:")
    print(df[column].value_counts())
    
    x = df[column].value_counts()
    
    # TAX_GRADE 전용 라벨
    if column == 'TAX_GRADE':
        labels = ['HIGH', 'MEDIUM', 'LOW']
    else:
        labels = x.index.tolist()
    
    plt.figure(figsize=figsize)
    plt.pie(x, explode=(0.1, 0, 0) if len(x) >= 3 else None, 
            labels=labels, autopct='%1.1f%%')
    plt.title(f'{column} 파이차트')
    plt.show()

def box_plot(column='TAX', figsize=(8, 6)):
    """단일 변수 상자그림"""
    print(f"=== {column} 상자그림 ===")
    
    print(f"{column} 기술통계:")
    print(df[column].describe())
    
    plt.figure(figsize=figsize)
    plt.boxplot([df[column]])
    plt.ylabel(column)
    plt.title(f'{column} 상자그림')
    plt.show()

def grouped_boxplot(x='CMEDV', y='TOWN', figsize=(12, 10)):
    """그룹별 상자그림"""
    print(f"=== {y}별 {x} 상자그림 ===")
    
    plt.figure(figsize=figsize)
    sns.boxplot(x=x, y=y, data=df)
    plt.title(f'{y}별 {x} 분포')
    plt.tight_layout()
    plt.show()

def hue_boxplot(x='CMEDV', y='TOWN', hue='TAX_GRADE', figsize=(14, 10)):
    """Hue가 있는 상자그림"""
    print(f"=== {y}별 {x} 분포 ({hue}로 구분) ===")
    
    plt.figure(figsize=figsize)
    sns.boxplot(x=x, y=y, hue=hue, data=df)
    plt.title(f'{y}별 {x} 분포 ({hue}로 구분)')
    plt.tight_layout()
    plt.show()

def correlation_matrix(figsize=(12, 10)):
    """상관관계 히트맵"""
    print("=== 상관관계 매트릭스 ===")
    
    # 숫자형 컬럼만 선택
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    plt.figure(figsize=figsize)
    correlation = df[numeric_cols].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
    plt.title('변수간 상관관계')
    plt.tight_layout()
    plt.show()

def run_all_visualizations():
    """모든 시각화 실행"""
    print("모든 시각화를 순차적으로 실행합니다...\n")
    
    data_overview()
    print("\n" + "="*50 + "\n")
    
    histogram()
    print("\n" + "="*30 + "\n")
    
    scatter_plot()
    print("\n" + "="*30 + "\n")
    
    pie_chart()
    print("\n" + "="*30 + "\n")
    
    box_plot()
    print("\n" + "="*30 + "\n")
    
    grouped_boxplot()
    print("\n" + "="*30 + "\n")
    
    hue_boxplot()
    print("\n" + "="*30 + "\n")
    
    correlation_matrix()

if __name__ == '__main__':
    fire.Fire()
