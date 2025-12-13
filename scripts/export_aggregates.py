"""Export aggregate CSVs for D3 dashboard
Reads data/raw/Telco_Customer_Churn_Dataset.csv and writes pre-aggregated CSVs to d3_visualizations/data/
"""
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
RAW = os.path.join(ROOT, 'data', 'raw', 'Telco_Customer_Churn_Dataset.csv')
OUT_DIR = os.path.join(ROOT, 'd3_visualizations', 'data')
os.makedirs(OUT_DIR, exist_ok=True)

print('Loading raw data from', RAW)
df = pd.read_csv(RAW)

# Ensure numeric
df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce').fillna(0).astype(int)
df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce').fillna(0.0)
# Standardise churn flag
df['ChurnFlag'] = df['Churn'].apply(lambda x: 1 if str(x).strip().lower()=='yes' else 0)

# 1. churn_distribution.csv
churned = int(df['ChurnFlag'].sum())
retained = int(len(df) - churned)
dist = pd.DataFrame([{'label':'Retained','value':retained,'color':'#4ecdc4'},
                     {'label':'Churned','value':churned,'color':'#ff6b6b'}])
dist.to_csv(os.path.join(OUT_DIR,'churn_distribution.csv'), index=False)
print('Wrote churn_distribution.csv')

# 2. contract_distribution.csv
contract = df.groupby('Contract').agg(
    total_customers=('customerID','count'),
    churned=('ChurnFlag','sum')
).reset_index()
contract['retained'] = contract['total_customers'] - contract['churned']
contract['churnRate'] = (contract['churned'] / contract['total_customers'] * 100).round(1)
contract = contract.rename(columns={'Contract':'contract'})
contract.to_csv(os.path.join(OUT_DIR,'contract_distribution.csv'), index=False)
print('Wrote contract_distribution.csv')

# 3. tenure_buckets.csv
buckets = [ (0,12,'0-12'), (13,24,'13-24'), (25,48,'25-48'), (49,60,'49-60'), (61,9999,'61-72') ]
rows = []
for lo,hi,label in buckets:
    sel = df[(df['tenure']>=lo) & (df['tenure']<=hi)]
    customers = len(sel)
    churned = int(sel['ChurnFlag'].sum())
    churnRate = round((churned/customers*100) if customers>0 else 0,1)
    rows.append({'tenure':label,'customers':customers,'churnRate':churnRate})

pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR,'tenure_buckets.csv'), index=False)
print('Wrote tenure_buckets.csv')

# 4. payment_methods.csv
pay = df.groupby('PaymentMethod').agg(total_customers=('customerID','count'), churned=('ChurnFlag','sum')).reset_index()
pay['retained'] = pay['total_customers'] - pay['churned']
pay['churnRate'] = (pay['churned']/pay['total_customers']*100).round(1)
pay = pay.rename(columns={'PaymentMethod':'method'})
pay.to_csv(os.path.join(OUT_DIR,'payment_methods.csv'), index=False)
print('Wrote payment_methods.csv')

# 5. internet_service.csv
inet = df.groupby('InternetService').agg(total_customers=('customerID','count'), churned=('ChurnFlag','sum')).reset_index()
inet['retained'] = inet['total_customers'] - inet['churned']
inet['color'] = inet['InternetService'].apply(lambda s: '#ff6b6b' if 'Fiber' in str(s) else ('#4ecdc4' if 'DSL' in str(s) else '#ffd93d'))
inet['churnRate'] = (inet['churned']/inet['total_customers']*100).round(1)
inet = inet.rename(columns={'InternetService':'service'})
inet.to_csv(os.path.join(OUT_DIR,'internet_service.csv'), index=False)
print('Wrote internet_service.csv')

# 6. monthly_charges_bins.csv (5 bins)
charges = df['MonthlyCharges'].dropna()
bins = pd.cut(charges, bins=5)
bin_df = df.copy()
bin_df['bin'] = pd.cut(bin_df['MonthlyCharges'], bins=5)
out_rows = []
for interval, group in bin_df.groupby('bin', observed=False):
    if pd.isna(interval):
        continue
    customers = len(group)
    churned = int(group['ChurnFlag'].sum())
    churnRate = round((churned/customers*100) if customers>0 else 0,1)
    out_rows.append({'range':f"{interval.left:.2f}-{interval.right:.2f}", 'count':customers, 'avgChurn':churnRate})

pd.DataFrame(out_rows).to_csv(os.path.join(OUT_DIR,'monthly_charges_bins.csv'), index=False)
print('Wrote monthly_charges_bins.csv')

print('All aggregates exported to', OUT_DIR)
