import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    df = pd.read_csv('data/raw/Telco_Customer_Churn_Dataset.csv')
    df = df[df['TotalCharges'] != ' ']
    for c in ['tenure','MonthlyCharges','TotalCharges']:
        df[c]=pd.to_numeric(df[c])
    X = df[['tenure','MonthlyCharges','TotalCharges']].copy()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=4, random_state=42, n_init=25).fit(Xs)
    df['cluster']=km.labels_+1
    summary = df.groupby('cluster').agg(
        n=('customerID','count'),
        avg_tenure=('tenure','mean'),
        avg_monthly=('MonthlyCharges','mean'),
        churned=('Churn', lambda x: (x=='Yes').sum()),
    )
    summary['churn_rate']=summary['churned']/summary['n']*100
    print(summary[['n','avg_tenure','avg_monthly','churn_rate']].round(2))
    print('\nTotal rows:', len(df))
