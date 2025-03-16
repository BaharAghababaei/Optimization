from gurobipy import Model,GRB,quicksum

def solve_opt_model(tr,org_params):
    
    P, F, I, T = org_params['P'], org_params['F'], org_params['I'], org_params['T']
    demand, total, cap = org_params['demand'], org_params['total'], org_params['cap']
    sh, h, v, = org_params['sh'], org_params['h'], org_params['v']

    mdl=Model('OPTIMAL MODEL')
    x=mdl.addVars(F,P,T, vtype=GRB.INTEGER, lb=0, name='x') # quantity assigned to each fulfilment center
    y=mdl.addVars(F,P,I,T, vtype=GRB.INTEGER,lb=0, name='y') # quantity shipped from fulfiment centers to regions
    Inv_1=mdl.addVars(F,P,T, vtype=GRB.INTEGER,lb=0, name='Inv_1')  # inventory level
    
    mdl.addConstrs((quicksum(y[f,p,i,t] for f in F)>=demand[p,i,t]
                         for i in I for p in P for t in T));     # demand satisfaction constraint
    
    mdl.addConstrs((Inv_1[f,p,t]==x[f,p,t]-quicksum(y[f,p,i,t] for i in I) for f in F for p in P for t in T if t ==1)); #Inventory balance equation
    mdl.addConstrs((Inv_1[f,p,t]==Inv_1[f,p,t-1]+x[f,p,t]-quicksum(y[f,p,i,t] for i in I) for f in F for p in P for t in T if t > 1)); #Inventory balance equation
    
    
    mdl.addConstrs((quicksum(x[f,p,t] for f in F)<=total[p,t]
                        for p in P for t in T)); 
    
    mdl.addConstrs((quicksum(v[p]*Inv_1[f,p,t-1] for p in P)+quicksum(v[p]*x[f,p,t] for p in P) <=cap[f]
                          for f in F for t in T if t>1)); 
    
    mdl.addConstrs((quicksum(v[p]*x[f,p,t] for p in P) <=cap[f]
                          for f in F for t in T if t==1));
    
    obj=quicksum(sh[f,p,t]*x[f,p,t] for f in F for p in P for t in T)+quicksum(h[f,p,t]*Inv_1[f,p,t] for f in F for p in P for t in T)+quicksum(tr[f,i,p]*y[f,p,i,t] for f in F for i in I for p in P for t in T)
    mdl.setObjective(obj, GRB.MINIMIZE)
    

    mdl.setParam('OutputFlag', 0)
    mdl.optimize()
    opt_obj=mdl.objval
    print('the optimal objective value:%f' % opt_obj)
    return opt_obj

  
    

    


