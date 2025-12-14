import pandas as pd

df = pd.read_csv('data/raw/Telco_Customer_Churn_Dataset.csv')
df = df[df['TotalCharges'] != ' ']

print('='*60)
print('OVERALL METRICS:')
print('='*60)
print(f'Total Customers: {len(df)}')
print(f'Churn Rate: {(len(df[df["Churn"]=="Yes"])/len(df)*100):.2f}%')
print(f'Avg Monthly Revenue: ${df["MonthlyCharges"].astype(float).mean():.2f}')
print(f'High Risk (Churned): {len(df[df["Churn"]=="Yes"])}')

print('\n' + '='*60)
print('CONTRACT TYPE:')
print('='*60)
for contract in sorted(df['Contract'].unique()):
    subset = df[df['Contract']==contract]
    churned = len(subset[subset['Churn']=='Yes'])
    total = len(subset)
    rate = (churned/total*100) if total>0 else 0
    print(f'{contract:20s}: {total:4d} customers, {churned:4d} churned, {rate:5.1f}% rate')

print('\n' + '='*60)
print('PAYMENT METHOD:')
print('='*60)
for payment in sorted(df['PaymentMethod'].unique()):
    subset = df[df['PaymentMethod']==payment]
    churned = len(subset[subset['Churn']=='Yes'])
    total = len(subset)
    rate = (churned/total*100) if total>0 else 0
    print(f'{payment:30s}: {total:4d} customers, {churned:4d} churned, {rate:5.1f}% rate')

print('\n' + '='*60)
print('INTERNET SERVICE:')
print('='*60)
for internet in sorted(df['InternetService'].unique()):
    subset = df[df['InternetService']==internet]
    churned = len(subset[subset['Churn']=='Yes'])
    total = len(subset)
    rate = (churned/total*100) if total>0 else 0
    print(f'{internet:20s}: {total:4d} customers, {churned:4d} churned, {rate:5.1f}% rate')

print('\n' + '='*60)
print('TENURE BUCKETS:')
print('='*60)
def get_tenure_bucket(t):
    if t <= 12: return '0-12 months'
    elif t <= 24: return '13-24 months'
    elif t <= 36: return '25-36 months'
    elif t <= 48: return '37-48 months'
    else: return '49+ months'

df['TenureBucket'] = df['tenure'].apply(get_tenure_bucket)
for bucket in ['0-12 months', '13-24 months', '25-36 months', '37-48 months', '49+ months']:
    subset = df[df['TenureBucket']==bucket]
    churned = len(subset[subset['Churn']=='Yes'])
    total = len(subset)
    rate = (churned/total*100) if total>0 else 0
    print(f'{bucket:20s}: {total:4d} customers, {churned:4d} churned, {rate:5.1f}% rate')

print('\n' + '='*60)
print('MONTHLY CHARGES BUCKETS:')
print('='*60)
def get_charge_bucket(c):
    if c < 30: return '$0-30'
    elif c < 60: return '$30-60'
    elif c < 90: return '$60-90'
    else: return '$90+'

df['ChargeBucket'] = df['MonthlyCharges'].astype(float).apply(get_charge_bucket)
for bucket in ['$0-30', '$30-60', '$60-90', '$90+']:
    subset = df[df['ChargeBucket']==bucket]
    churned = len(subset[subset['Churn']=='Yes'])
    total = len(subset)
    rate = (churned/total*100) if total>0 else 0
    print(f'{bucket:20s}: {total:4d} customers, {churned:4d} churned, {rate:5.1f}% rate')
