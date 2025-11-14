import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(report_date: str):
    """โหลดข้อมูลตามวันที่รายงาน"""
    print(f"Loading data for {report_date}...")
    df_tran = pd.read_csv(f'Transection_{report_date}.csv', sep='|', encoding='utf-8-sig')
    df_perf = pd.read_excel('Performance.xlsx', sheet_name='Sheet1')
    
    # Strip columns เพื่อให้ชื่อคอลัมน์สะอาด
    df_tran.columns = df_tran.columns.str.strip()
    df_perf.columns = df_perf.columns.str.strip()
    
    return df_tran, df_perf

def clean_data(df_tran, df_perf):
    """ทำความสะอาดข้อมูล"""
    print("Cleaning data...")

    # แปลงวันที่
    df_tran['FRPDATE'] = pd.to_datetime(df_tran['FRPDATE'], format='%Y%m%d', errors='coerce')
    df_tran['FNPLFDTE'] = pd.to_datetime(df_tran['FNPLFDTE'], format='%Y%m%d', errors='coerce')
    df_tran['FORDATE'] = pd.to_datetime(df_tran['FORDATE'], format='%Y%m%d', errors='coerce')
    df_tran['FMATDATE'] = pd.to_datetime(df_tran['FMATDATE'], format='%Y%m%d', errors='coerce')

    df_tran.drop_duplicates(inplace=True)

    # Merge ข้อมูล
    df_merged = pd.merge(df_tran, df_perf, left_on='FCUSNO', right_on='CIF', how='left')

    # กรณี merge แล้วมี column ซ้ำ
    if 'FRPDATE_x' in df_merged.columns:
        df_merged.rename(columns={'FRPDATE_x': 'FRPDATE'}, inplace=True)
    if 'FRPDATE_y' in df_merged.columns:
        df_merged.drop(columns=['FRPDATE_y'], inplace=True)

    # เติมค่า missing
    if 'FDPDUE00' in df_merged.columns:
        df_merged['FDPDUE00'] = pd.to_numeric(df_merged['FDPDUE00'], errors='coerce').fillna(0)
    else:
        df_merged['FDPDUE00'] = 0

    return df_merged

def feature_engineer(df_merged):
    """สร้างตัวแปรใหม่"""
    print("Engineering features...")

    # Map Stage
    stage_map = {1: '1. Performing', 2: '2. Under-performing', 3: '3. NPL'}
    df_merged['Stage_Name'] = df_merged['STAGE_CIF'].map(stage_map).fillna('0. Unknown')

    # Overdue
    df_merged['Is_Overdue'] = df_merged['FDPDUE00'] > 0

    # DPD Bucket
    bins = [-1, 0, 30, 60, 90, float('inf')]
    labels = ['0. No DPD','1. 1-30 Days','2. 31-60 Days','3. 61-90 Days','4. 90+ Days']
    df_merged['DPD_Bucket'] = pd.cut(df_merged['FDPDUE00'], bins=bins, labels=labels, right=True)

    # แปลงเป็น numeric สำหรับคำนวณ
    df_merged['FFLGBWFW'] = pd.to_numeric(df_merged['FFLGBWFW'], errors='coerce')
    df_merged['FPRINCAM'] = pd.to_numeric(df_merged['FPRINCAM'], errors='coerce')

    # Debt to Limit Ratio
    df_merged['Debt_to_Limit_Ratio'] = np.where(
        df_merged['FFLGBWFW'] > 0,
        df_merged['FPRINCAM'] / df_merged['FFLGBWFW'],
        np.nan
    )

    # อายุสินเชื่อและ tenor
    df_merged['Loan_Age_Days'] = (df_merged['FRPDATE'] - df_merged['FORDATE']).dt.days
    df_merged['Remaining_Tenor_Days'] = (df_merged['FMATDATE'] - df_merged['FRPDATE']).dt.days
    df_merged['Loan_Orig_Year'] = df_merged['FORDATE'].dt.year
    df_merged['Loan_Orig_Quarter'] = df_merged['FORDATE'].dt.quarter

    return df_merged

def create_report(df_merged, report_date: str):
    """สร้างรายงานและ Dashboard"""
    print("Creating report and dashboard...")

    report_by_stage = df_merged.groupby('Stage_Name')['FPRINCAM'].agg(['sum', 'mean', 'count'])

    sns.set_theme(style="darkgrid", palette="Blues_d", font="Tahoma")

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(f'Portfolio Health Dashboard ({report_date})', fontsize=22, fontweight='bold', y=0.98)

    # Pie Chart
    colors = sns.color_palette('pastel')[0:len(report_by_stage)]
    axes[0, 0].pie(report_by_stage['count'], labels=report_by_stage.index,
                   autopct='%1.1f%%', startangle=90, colors=colors)
    axes[0, 0].set_title('Account Count by Stage', fontsize=14, fontweight='bold')

    # Bar Chart
    sns.barplot(ax=axes[0, 1], data=report_by_stage.reset_index(),
                x='Stage_Name', y='sum', palette='viridis')
    axes[0, 1].set_title('Total Principal by Stage', fontsize=14, fontweight='bold')

    # Box Plot
    sns.boxplot(ax=axes[1, 0], data=df_merged, x='FPRODTY', y='FPRINCAM', palette='Set2')
    axes[1, 0].set_title('Principal Distribution by Product', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylim(0, 12000000)

    # Histogram
    df_npl_90 = df_merged[df_merged['FDPDUE00'] > 90]
    sns.histplot(ax=axes[1, 1], data=df_npl_90, x='FDPDUE00',
                 bins=20, kde=True, color='crimson', alpha=0.7)
    axes[1, 1].set_title('DPD Distribution (90+ Days)', fontsize=14, fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(f'monthly_dashboard_{report_date}.png', dpi=300, bbox_inches='tight')
    print(f"Report for {report_date} created successfully!")

def main(report_date: str):
    """ฟังก์ชันหลัก"""
    df_tran, df_perf = load_data(report_date)
    df_merged = clean_data(df_tran, df_perf)
    df_final = feature_engineer(df_merged)
    create_report(df_final, report_date)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run NPL Report Pipeline')
    parser.add_argument('--date', required=True, help='Report date in YYYYMMDD format')
    args = parser.parse_args()
    main(args.date)
