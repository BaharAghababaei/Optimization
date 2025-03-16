import numpy as np
import random
import pandas as pd
import math



random.seed(0)
np.random.seed(0)

def load_dataset():
    T = list(range(1, 5))#periods

    P = list(range(1, 101)) # items
    
    # demand points locations
    dp = pd.read_csv('C:/Users/bahar/Documents/RA/Amazon_network_planning/US_counties.csv')
    dp.columns = dp.columns.str.strip()
    dp = dp[['ANSICODE', 'INTPTLAT', 'INTPTLONG', 'population']].drop_duplicates(subset=['ANSICODE'])
    
    dp['pop_ratio'] = dp['population'] / dp['population'].sum()
    dp['county_magnitude'] = (10000 * dp['pop_ratio']).round()
    dp = dp[dp['county_magnitude'] >= 10] 
    dp = dp[~((dp['INTPTLAT'] < 25) & (dp['INTPTLONG'] < -155))]
    dp = dp[['ANSICODE', 'INTPTLAT', 'INTPTLONG', 'pop_ratio']]
    I = list(dp['ANSICODE'])
    
    #warehouses
    df = pd.read_csv('C:/Users/bahar/Documents/RA/Amazon_network_planning/FC_list.csv')
    df = df[df['Accuracy Score'] != 0][['Code', 'Latitude', 'Longitude']].drop_duplicates(subset=['Code'])
    F = list(random.sample(list(df['Code']), 50))
    df = df[df['Code'].isin(F)]
    #demand
    demand_total = {(p, t): round(np.random.uniform(10000, 20000)) for p in P for t in T}
    dp_dict = dp.set_index('ANSICODE')['pop_ratio'].to_dict()  
    demand = {
        (p, i, t): math.floor(demand_total[p, t] * dp_dict[i])  
        for p in P for t in T for i in I}
    supply_total = {p: round(1.5 * sum(demand_total[p, t] for t in T)) for p in P}
    
    total = {
        (p, t): round(np.random.normal(supply_total[p] / len(T), 1000 if t > 1 else 0))
        for p in P for t in T}
    
    sum_demand={p: sum(demand_total[p,t] for t in T) for p in P}
    h={(f,p,t): np.random.uniform(1,5) for f in F for p in P for t in T} # storage cost
    sh={(f,p,t): np.random.uniform(1,6) for f in F for p in P for t in T} # Procurement cost
    v={p: np.random.randint(1,35) for p in P} #volume
    sum_cap=math.ceil(sum(sum_demand[p]*(v[p]) for p in P ))
    sqaure_footage={f: np.random.randint(600000,1000001) for f in F} #square feet
    cap={f: round(sum_cap*(sqaure_footage[f]/sum(sqaure_footage[f]for f in F))) for f in F}
    
   
    return  {
        'P': P, 'F': F, 'I': I, 'dp': dp, 'df': df, 'T': T, 
        'demand': demand, 'sum_demand': sum_demand, 'total': total, 
        'cap': cap, 'sh': sh, 'h': h, 'v': v}


def load_agg_params(org_params,tr, W, K, C, w_F, I_k, p_c, cluster_counts_d, cluster_counts_f, cluster_counts_p):

    F, I, T = (org_params["F"], org_params["I"],org_params["T"])
    demand, total, cap, v, h, sh= (org_params["demand"], org_params["total"], org_params["cap"],
                                         org_params["v"], org_params["h"], org_params["sh"])
    agg_params = {
        "demand_agg": {(c, k, t): sum(demand[p, i, t] for p in p_c[c] for i in I_k[k]) for c in C for k in K   for t in T},
        "total_agg": {(c, t): sum(total[p, t] for p in p_c[c]) for c in C for t in T},
        "cap_agg": {w: sum(cap[f] for f in w_F[w]) for w in W},
        "v_agg": {c: math.ceil(sum(v[p] for p in p_c[c]) / cluster_counts_p[c]) for c in C},
        "h_agg": {(w, c, t): sum(h[f, p, t] for p in p_c[c] for f in w_F[w]) / 
                  (cluster_counts_f[w] * cluster_counts_p[c]) for w in W for c in C for t in T},
        "sh_agg": {(w, c, t): sum(sh[f, p, t] for p in p_c[c] for f in w_F[w]) / 
                   (cluster_counts_f[w] * cluster_counts_p[c]) for w in W for c in C for t in T},
        "tr_agg": {(w, k, c): sum(tr[f, i, p] for f in w_F[w] for i in I_k[k] for p in p_c[c]) / 
                   (cluster_counts_d[k] * cluster_counts_f[w] * cluster_counts_p[c]) for w in W for k in K for c in C},
        "tr_k": {k: {(w, i, c): sum(tr[f, i, p] for f in w_F[w] for p in p_c[c]) /
                     (cluster_counts_f[w] * cluster_counts_p[c]) for w in W for i in I_k[k] for c in C} for k in K},
        "demand_k": {k: {(c, i, t): sum(demand[p, i, t] for p in p_c[c]) for c in C for i in I_k[k] for t in T} for k in K},
        "h_w": {w: {(f, c, t): sum(h[f, p, t] for p in p_c[c]) / cluster_counts_p[c]
                     for f in w_F[w] for c in C for t in T} for w in W},
        "sh_w": {w: {(f, c, t): sum(sh[f, p, t] for p in p_c[c]) / cluster_counts_p[c]
                      for f in w_F[w] for c in C for t in T} for w in W},
        "tr_w": {w: {(f, i, c): sum(tr[f, i, p] for p in p_c[c]) / cluster_counts_p[c]
                      for f in w_F[w] for i in I for c in C} for w in W},
        "h_c": {c: {(f, p, t): h[f, p, t] for f in F for p in p_c[c] for t in T} for c in C},
        "sh_c": {c: {(f, p, t): sh[f, p, t] for f in F for p in p_c[c] for t in T} for c in C},
        "tr_c": {c: {(f, i, p): tr[f, i, p] for f in F for p in p_c[c] for i in I} for c in C},
        "demandp": {c: {(i, p, t): demand[p, i, t] for i in I for p in p_c[c] for t in T} for c in C}
    }

    return agg_params

'''
     
    tr_agg = {(w, k, c): sum(tr[f,i,p] for f in w_F[w] for i in I_k[k] for p in p_c[c]) /
          (cluster_counts_d[k] * cluster_counts_f[w] * cluster_counts_p[c])  for w in W for k in K for c in C}
    total_agg = {(c, t): sum(total[p, t] for p in p_c[c]) for c in C for t in T}
    h_agg={(w,c,t):sum(h[f,p,t] for p in p_c[c] for f in w_F[w])/(cluster_counts_f[w] * cluster_counts_p[c]) for w in W for c in C for t in T} # storage cost
    sh_agg={(w,c,t):sum(sh[f,p,t] for p in p_c[c] for f in w_F[w])/(cluster_counts_f[w] * cluster_counts_p[c]) for w in W for c in C for t in    T}
    cap_agg={w: sum(cap[f] for f in w_F[w])  for w in W}
    #v_agg={c: ((max(v[p] for p in p_c[c]))) for c in C} #volume
    v_agg={c: math.ceil((sum(v[p] for p in p_c[c])/cluster_counts_p[c])) for c in C}
    demand_agg={(c,k,t): sum(demand[p,i,t] for p in p_c[c] for i in I_k[k]) for c in C for k in K for t in T}
    tr_k = {k: {(w, i, c): sum(tr[f, i, p] for f in w_F[w] for p in p_c[c]) /(cluster_counts_f[w] * cluster_counts_p[c]) for w in W for i in I_k[k] for c in C} for k in K}
    demand_k = { k: {(c, i, t): sum(demand[p, i, t] for p in p_c[c]) for c in C for i in I_k[k] for t in T} for k in K}
    h_w = {w: {(f, c, t): sum(h[f, p, t] for p in p_c[c]) / cluster_counts_p[c]
            for f in w_F[w] for c in C for t in T}for w in W}
            
    sh_w = {w: {(f, c, t): sum(sh[f, p, t] for p in p_c[c]) / cluster_counts_p[c]
            for f in w_F[w] for c in C for t in T }for w in W}
    
    tr_w= {w: {(f, i, c): sum(tr[f, i, p] for p in p_c[c]) / cluster_counts_p[c]
            for f in w_F[w] for i in I for c in C}for w in W}
    sh_c={}
    for c in C:
        sh_c[c]={(f,p,t): sh[f, p, t] for f in F for p in p_c[c] for t in T}   
    h_c={}
    for c in C:
        h_c[c]={(f,p,t): h[f, p, t] for f in F for p in p_c[c] for t in T}
    
    tr_c={}
    for c in C:
        tr_c[c]={(f,i,p): tr[f, i,p] for f in F for p in p_c[c] for i in I}
    demandp={}
    for c in C:
        demandp[c]={(i,p,t): demand[p,i,t] for i in I for p in p_c[c] for t in T }
'''
  



