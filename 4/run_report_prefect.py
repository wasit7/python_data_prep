# run_report_prefect_fixed.py - NPL Report with Prefect
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prefect import task, flow, get_run_logger

# ---------------- Tasks ---------------- #
@task(retries=3, retry_delay_seconds=10)
def load_data(report_date: str):
    logger = get_run_logger()
    logger.info(f"Loading data for {report_date} ...")
    
    df_tran = pd.read_csv(f'Transection_{report_date}.csv', sep='|', encoding='utf-8-sig')
    df_perf = pd.read_excel('Performance.xlsx', sheet_name='Sheet1')
    
    # Strip columns
    df_tran.columns = df_tran.columns.str.strip()
    df_perf.columns = df_perf.columns.str.strip()
    
    logger.info(f"Loaded {len(df_tran):,} transactions")
    return df_tran, df_perf

@task
def clean_data(df_tran, df_perf):
    logger = get_run_logger()
    logger.info("Cleaning data ...")
    
    # แปลงวันที่
    for col in ['FRPDATE', 'FNPLFDTE', 'FORDATE', 'FMATDATE']:
        if col in df_tran.columns:
            df_tran[col] = pd.to_datetime(df_tran[col], format='%Y%m%d', errors='coerce')
    
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
    
    logger.info(f"Cleaned {len(df_merged):,} records")
    return df_merged

@task
def feature_engineer(df_merged):
    logger = get_run_logger()
    logger.info("Engineering features ...")
    
    # Stage mapping
    stage_map = {1: '1. Performing', 2: '2. Under-performing', 3: '3. NPL'}
    df_merged['Stage_Name'] = df_merged['STAGE_CIF'].map(stage_map).fillna('0. Unknown')
    
    df_merged['Is_Overdue'] = df_merged['FDPDUE00'] > 0
    
    bins = [-1, 0, 30, 60, 90, float('inf')]
    labels = ['0. No DPD','1. 1-30 Days','2. 31-60 Days','3. 61-90 Days','4. 90+ Days']
    df_merged['DPD_Bucket'] = pd.cut(df_merged['FDPDUE00'], bins=bins, labels=labels, right=True)
    
    df_merged['FFLGBWFW'] = pd.to_numeric(df_merged['FFLGBWFW'], errors='coerce')
    df_merged['FPRINCAM'] = pd.to_numeric(df_merged['FPRINCAM'], errors='coerce')
    
    df_merged['Debt_to_Limit_Ratio'] = np.where(
        df_merged['FFLGBWFW'] > 0,
        df_merged['FPRINCAM'] / df_merged['FFLGBWFW'],
        np.nan
    )
    
    df_merged['Loan_Age_Days'] = (df_merged['FRPDATE'] - df_merged['FORDATE']).dt.days
    df_merged['Remaining_Tenor_Days'] = (df_merged['FMATDATE'] - df_merged['FRPDATE']).dt.days
    df_merged['Loan_Orig_Year'] = df_merged['FORDATE'].dt.year
    df_merged['Loan_Orig_Quarter'] = df_merged['FORDATE'].dt.quarter
    
    logger.info("Features created successfully")
    return df_merged

@task
def create_report(df_final, report_date: str):
    logger = get_run_logger()
    logger.info("Creating dashboard report ...")
    
    report_by_stage = df_final.groupby('Stage_Name')['FPRINCAM'].agg(['sum', 'mean', 'count'])
    
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
    sns.boxplot(ax=axes[1, 0], data=df_final, x='FPRODTY', y='FPRINCAM', palette='Set2')
    axes[1, 0].set_title('Principal Distribution by Product', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylim(0, 12000000)
    
    # Histogram
    df_npl_90 = df_final[df_final['FDPDUE00'] > 90]
    sns.histplot(ax=axes[1, 1], data=df_npl_90, x='FDPDUE00',
                 bins=20, kde=True, color='crimson', alpha=0.7)
    axes[1, 1].set_title('DPD Distribution (90+ Days)', fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    output_file = f'monthly_dashboard_{report_date}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Dashboard saved as {output_file}")

# ---------------- Flow ---------------- #
@flow(name="NPL Report Flow")
def run_report_flow(report_date: str):
    df_tran, df_perf = load_data(report_date)
    df_merged = clean_data(df_tran, df_perf)
    df_final = feature_engineer(df_merged)
    create_report(df_final, report_date)
    print(f"Flow completed for {report_date}")

# ---------------- Run ---------------- #
if __name__ == "__main__":
    import sys
    report_date = sys.argv[1] if len(sys.argv) > 1 else "20240731"
    run_report_flow(report_date)
