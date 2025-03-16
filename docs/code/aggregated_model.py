from gurobipy import Model, GRB, quicksum

def solve_aggregated_model(W, K, C, T, agg_params):

    demand_agg, total_agg, cap_agg = agg_params["demand_agg"], agg_params["total_agg"], agg_params["cap_agg"]
    v_agg, h_agg, sh_agg, tr_agg = agg_params["v_agg"], agg_params["h_agg"], agg_params["sh_agg"],  agg_params["tr_agg"]


    mdlg=Model('aggregation model')
   
    x=mdlg.addVars(W,C,T, vtype=GRB.INTEGER, lb=0, name='x') # quantity assigned to each fulfilment center
    y=mdlg.addVars(W,K,C,T, vtype=GRB.INTEGER,lb=0, name='y') # quantity shipped from fulfiment centers to regions
    Inv_1=mdlg.addVars(W,C,T, vtype=GRB.INTEGER,lb=0, name='Inv_1')  # inventory level
    
    
    mdlg.addConstrs((quicksum(y[w,k,c,t] for w in W)>=demand_agg[c,k,t]
                          for k in K for c in C for t in T));
    
    mdlg.addConstrs((Inv_1[w,c,t]==x[w,c,t]-quicksum(y[w,k,c,t] for k in K) for w in W for c in C  for t in T if t ==1)); #Inventory balance equation
    mdlg.addConstrs((Inv_1[w,c,t]==Inv_1[w,c,t-1]+x[w,c,t]-quicksum(y[w,k,c,t] for k in K) for w in W for c in C for t in T if t > 1)); #Inventory balance equation
    
    
    
    mdlg.addConstrs((quicksum(x[w,c,t] for w in W)<=total_agg[c,t]
                        for c in C for t in T));
    
    mdlg.addConstrs((quicksum(v_agg[c]*x[w,c,t] for c in C) <=cap_agg[w]
                          for w in W for t in T if t==1));
    
    
    mdlg.addConstrs((quicksum(v_agg[c]*Inv_1[w,c,t-1] for c in C)+quicksum(v_agg[c]*x[w,c,t] for c in C) <=cap_agg[w]
                          for w in W for t in T if t>1));
    
    
    obj=quicksum(sh_agg[w,c,t]*x[w,c,t] for w in W for c in C for t in T)+quicksum(h_agg[w,c,t]*Inv_1[w,c,t] for w in W for c in C for t in T)+quicksum(tr_agg[w,k,c]*y[w,k,c,t] for w in W for k in K for c in C for t in T)
    mdlg.setObjective(obj, GRB.MINIMIZE)
    mdlg.setParam('OutputFlag', 0)
    mdlg.optimize()
    print('AGG objective function value:%f' % mdlg.objval)
    x_agg = {key: var.x for key, var in x.items() }
    y_agg = {key: var.x for key, var in y.items() }
    Inv_agg = {key: var.x for key, var in Inv_1.items() }
    
    y_bar_k = {k: {(w, c, t): y_agg[w, k, c, t] for w in W for c in C for t in T } for k in K}
    
    W_k = {k: [w for w in W if any(y_bar_k[k][w, c, t] > 0 for c in C for t in T)] for k in K}
    x_bar_w = {w: {(c, t): x_agg.get((w, c, t), 0)
        for c in C for t in T}for w in W}
        
    inv_bar_w = {w: {(c, t): Inv_agg.get((w, c, t), 0)
        for c in C for t in T}for w in W}
   
    agg_results = {
        "x_agg": x_agg,
        "y_agg": y_agg,
        "Inv_agg": Inv_agg,
        "y_bar_k": y_bar_k,
        "W_k": W_k,
        "x_bar_w": x_bar_w,
        "inv_bar_w": inv_bar_w}


    return agg_results
	
