from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


import numpy as np
import pandas as pd



def cluster_demand_points(params_d):
    
    dp,nd=params_d['dp'],params_d['nd']

    dp_reduced = dp[['ANSICODE', 'INTPTLAT', 'INTPTLONG','pop_ratio']].copy()

    scaler = StandardScaler()
    dp_reduced[['INTPTLAT_N', 'INTPTLONG_N']] = scaler.fit_transform(dp_reduced[['INTPTLAT', 'INTPTLONG']])

    kmeans = KMeans(n_clusters=nd, init='k-means++', random_state=20, n_init=10) #try diiferent n_init 10 or 20
    dp_reduced['cluster_label'] = kmeans.fit_predict(dp_reduced[['INTPTLAT_N', 'INTPTLONG_N']])

    dp_reduced.drop(columns=['INTPTLAT_N', 'INTPTLONG_N'], inplace=True)
    K=list(range(0, nd))
    I_k = {k: dp_reduced.loc[dp_reduced['cluster_label'] == k, 'ANSICODE'].tolist() for k in K}
    cluster_counts_d = dp_reduced['cluster_label'].value_counts().to_dict()
    return dp_reduced, I_k,K,cluster_counts_d



def cluster_warehouses(params_w):
   
    df,nf=params_w['df'],params_w['nf']	

    df_reduced = df[['Code', 'Latitude', 'Longitude']].copy()

    scaler = StandardScaler()
    df_reduced[['Latitude_N', 'Longitude_N']] = scaler.fit_transform(df_reduced[['Latitude', 'Longitude']])

    kmeans = KMeans(n_clusters=nf, init='k-means++', random_state=20, n_init=10) #try diiferent n_init
    df_reduced['cluster_label'] = kmeans.fit_predict(df_reduced[['Latitude_N', 'Longitude_N']])

    df_reduced.drop(columns=['Latitude_N', 'Longitude_N'], inplace=True)
    W=list(range(0, nf))
    w_F = {w: df_reduced.loc[df_reduced['cluster_label'] == w, 'Code'].tolist() for w in range(nf)}
    cluster_counts_f = df_reduced['cluster_label'].value_counts().to_dict()
    return df_reduced, w_F,W,cluster_counts_f



def cluster_items(params_p):

    P, F, T = params_p['P'], params_p['F'], params_p['T']
    h, sh, v = params_p['h'], params_p['sh'], params_p['v']    
    sum_demand, n_clusters = params_p['sum_demand'],params_p['n_clusters'] 
  
    avg_storage_costp = {p:    np.mean([h[f, p, t] for f in F for t in T]) for p in P}
    avg_receiving_costp = {p: np.mean([sh[f, p, t] for f in F for t in T]) for p in P}

    data = {
        'Volume': [v[p] for p in P],
        'Weight': [np.random.randint(1, 36) for p in P],  # Assuming weight needs to be defined
        'Avg_Receiving_Cost': [avg_receiving_costp[p] for p in P],
        'Avg_Storage_Cost': [avg_storage_costp[p] for p in P],
        'Total_Demand': [sum_demand[p] for p in P]
    }

    ddp = pd.DataFrame(data, index=P)

    scaler = StandardScaler()
    feature_cols = ['Volume', 'Weight','Avg_Receiving_Cost', 'Avg_Storage_Cost']
    ddp_scaled = scaler.fit_transform(ddp[feature_cols])
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=20, n_init=10)
    ddp['cluster_label'] = kmeans.fit_predict(ddp_scaled)
    C=list(range(0, n_clusters))
    p_c = {c: ddp[ddp['cluster_label'] == c].index.tolist() for c in range(n_clusters)}
    cluster_counts_p=ddp['cluster_label'].value_counts().to_dict()
    return p_c,C,cluster_counts_p

