from gurobipy import Model, GRB, quicksum
import concurrent.futures
import time
import math


def solve_disagg_stage1(W,K, C, I_k, org_params,agg_params,agg_results , max_workers=8):
    I,T= org_params["I"],org_params["T"]
    demand_k, tr_k = agg_params["demand_k"], agg_params["tr_k"]
    W_k,y_bar_k=agg_results['W_k'],agg_results['y_bar_k']
    
    def sub_k(k):
        W_kk, I_kk = W_k[k], I_k[k]
        demand_kk, y_bar_kk, tr_kk = demand_k[k], y_bar_k[k], tr_k[k]
        z1_values_k = {}
    
        mdl1_k = Model('sub_problem_k')
        z1 = mdl1_k.addVars(W_kk, I_kk, C, T, vtype=GRB.INTEGER, lb=0, name='z1')
    
    
        mdl1_k.addConstrs(
            (quicksum(z1[w, i, c, t] for w in W_kk) == demand_kk[c, i, t]
             for c in C for i in I_kk for t in T))
    
        mdl1_k.addConstrs(
            (quicksum(z1[w, i, c, t] for i in I_kk) <= y_bar_kk.get((w, c, t), 0)
             for w in W_kk for c in C for t in T))
    
        mdl1_k.setObjective(
            quicksum(tr_kk[w, i, c] * z1[w, i, c, t] for w in W_kk for i in I_kk for c in C for t in T),
            GRB.MINIMIZE)
        mdl1_k.setParam('OutputFlag', 0)
        mdl1_k.optimize()
    
        z1_values_k = {(w, i, c, t): var.x for (w, i, c, t), var in z1.items() if var.x > 0}
    
        return mdl1_k.objVal, z1_values_k
        

    start_time = time.time()
    
    z1_values, obj = {}, {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(sub_k, k): k for k in K}
    
        for future in concurrent.futures.as_completed(futures):
            k = futures[future]
            obj[k], z1_values[k] = future.result()
            
    
    z_bar_w={}
    for w in W:
       z_bar_w[w] = {(i, c, t): z1_values[k].get((w, i, c, t), 0)  for k in K for i in I_k[k] for c in C for t in T}   
    
    I_prime = {w: [i for i in I if any(z_bar_w[w][i, c, t] > 0 for c in C for t in T)] for w in W}
    new_demand_zone_numbers = { (i, c): idx for idx, (i, c) in enumerate({(i, c) for i in I for c in C}, start=1)}
    z_bar_w_phi = {w: {} for w in W}


    for w in W:
        for c in C:
            for i in I_prime[w]:
                for t in T:
                    phi = (i, c)
                    new_phi = new_demand_zone_numbers.get(phi)
                    if new_phi is not None:
                        z_bar_w_value = z_bar_w[w].get((i, c, t))
                        if z_bar_w_value is not None:
                            z_bar_w_phi[w][(new_phi, t)] = z_bar_w_value
    total_time = time.time() - start_time
    print(f" Stage 1 Disaggregation Completed in: {total_time:.2f} seconds")

    disagg_results_1 = {
        "obj": obj,
        "z1_values": z1_values,
        "z_bar_w": z_bar_w,
        "z_bar_w_phi": z_bar_w_phi,
        "I_prime": I_prime,
        "new_demand_zone_numbers": new_demand_zone_numbers
    }    


    return disagg_results_1

####stage 2

def sort_bundles_stage2(W, C, T, I_prime, sh_w, h_w, tr_w, w_F,z_bar_w_phi, v_agg):

    sorted_bundles_per_period = {w: [] for w in W}
    for w in W:
        for t in T:
            sorted_bundles = sorted(
                [(i, c) for c in C for i in I_prime[w]],
                key=lambda phi: (
                    (sum(
                        (sh_w[w][(f, phi[1], t)] +
                         h_w[w][(f, phi[1], t)] +
                         tr_w[w][(f, phi[0], phi[1])])
                        for f in w_F[w]
                    ) * z_bar_w_phi[w].get((phi[0], phi[1], t), 0))*v_agg[phi[1]] # Ensure correct indices for z_bar_w_phi
                ),
                reverse=True
            )
            sorted_bundles_per_period[w].append(sorted_bundles)
    return  sorted_bundles_per_period


def find_sorted_warehouses_for_bundle(w, i, c, t, w_bar_ft, w_F, cap1, sh_w, h_w, tr_w, v_agg):
    warehouse_costs = []

    for f in w_F[w] :

        if w_bar_ft[f, t] + v_agg[c] <= cap1[f, t]:
            if t >= 2:
                cost = (sh_w[w][(f, c, t)] + h_w[w][(f, c, t-1)] + tr_w[w][(f, i, c)])
                warehouse_costs.append((f, cost))
            else:
                cost = (sh_w[w][(f, c, t)] + h_w[w][(f, c, t)] + tr_w[w][(f, i, c)])
                warehouse_costs.append((f, cost))
    sorted_warehouses= sorted(warehouse_costs, key=lambda x: x[1])
    return [f for f, _ in sorted_warehouses]


def assign_bundles(w, T, C, z_bar_w_phi, x_bar_w, cap1, w_F, v_agg, I_prime, sh_w, h_w, tr_w, new_demand_zone_numbers,sorted_bundles_per_period):

    demand_phi_copy = z_bar_w_phi[w].copy()
    total_c=x_bar_w[w]
    w_bar_ft = {(f, t): 0 for f in w_F[w] for t in T}
    x_bar_ct = {(c, t): 0 for c in C for t in T}
    y_prime_ftc = {t: {} for t in T}
    x_prime_ft = {t: {} for t in T}
    Inv = {t: {} for f in w_F[w] for c in C for t in T}
    fc = {t: [] for t in T}
    remove_duplicate = {t: [] for t in T}
    f_star = {}


    for t in reversed(T):
        y_prime_ftc[t] = {}
        x_prime_ft[t] = {}

        for (i, c) in sorted_bundles_per_period[w][t-1]:
            phi = new_demand_zone_numbers[i, c]
            assigned = False
            offset_value = 0
            while demand_phi_copy.get((phi, t), 0) > 0:
                warehouses = find_sorted_warehouses_for_bundle(w, i, c, t, w_bar_ft, w_F, cap1, sh_w, h_w, tr_w, v_agg)
                original_demand = demand_phi_copy[phi, t]

                wrapped_warehouses = warehouses[offset_value:offset_value+1] + warehouses[:offset_value]+warehouses[offset_value+1:]

                for f_star in wrapped_warehouses:
                    if w_bar_ft[f_star, t] + v_agg[c] <= cap1[f_star, t]:
                        y_prime = math.floor(min(demand_phi_copy[phi, t], (cap1[f_star, t] - w_bar_ft[f_star, t]) / v_agg[c]))
                        y_prime_ftc[t][(f_star, i, c)] = y_prime
                        w_bar_ft[f_star, t] += v_agg[c] * y_prime
                        demand_phi_copy[phi, t] -= y_prime
                        if demand_phi_copy[phi, t] == 0:
                            assigned = True
                            break

                if assigned==True:
                    break
                elif  assigned==False:
                    for (f_star, _, _) in list(y_prime_ftc[t].keys()):
                        if (f_star, i, c) in y_prime_ftc[t]:
                            y_prime_value = y_prime_ftc[t][(f_star, i, c)]
                            w_bar_ft[f_star, t] -= v_agg[c] * y_prime_value
                            y_prime_value=0

                    demand_phi_copy[phi, t] = original_demand
                    if offset_value == len(warehouses):
                        break
                    else:
                        offset_value+=1

        remove_duplicate[t] = []

        fc[t] = sorted(
            [(f, c) for (f, i, c), value in y_prime_ftc[t].items()] +
            [(f, c) for (f, c), value in Inv[t].items() if value > 0],
            key=lambda x: Inv[t].get((x[0], x[1]), 0) + sum(y_prime_ftc[t].get((x[0],i,x[1]),0) for i in I_prime[w]),
            reverse=True
        )

        for (f, c) in fc[t]:
            if (f, c) not in remove_duplicate[t]:
                remove_duplicate[t].append((f, c))
                if t == 4:
                    if x_bar_ct[c, t] < total_c[c,t]:
                        x_prime = min((sum(y_prime_ftc[t].get((f, i, c), 0) for i in I_prime[w])), (total_c[c,t] - x_bar_ct[c, t]) )
                        if x_prime == sum(y_prime_ftc[t].get((f, i, c), 0) for i in I_prime[w]):
                            Inv_prime = 0
                        else:
                            Inv_prime = sum(y_prime_ftc[t].get((f, i, c), 0) for i in I_prime[w]) - x_prime
                    else:
                        x_prime = 0
                        Inv_prime = sum(y_prime_ftc[t].get((f, i, c), 0) for i in I_prime[w])
                    x_bar_ct[c, t] += x_prime
                    Inv[t-1][(f, c)] = Inv_prime
                    w_bar_ft[f, t-1] += v_agg[c] * Inv_prime
                 #   Inv_t_1[t - 1][(f, c)]=Inv[t-1].get((f, c),0)

                elif 1 < t < 4:
                    if  x_bar_ct[c, t] < total_c[c,t]:
                        x_prime = min((sum(y_prime_ftc[t].get((f, i, c), 0) for i in I_prime[w]) + Inv[t].get((f, c), 0)), (total_c[c,t] - x_bar_ct[c, t]))
                        if x_prime == sum(y_prime_ftc[t].get((f, i, c), 0) for i in I_prime[w]) + Inv[t].get((f, c), 0):
                            Inv_prime = 0
                        else:
                            Inv_prime = sum(y_prime_ftc[t].get((f, i, c), 0) for i in I_prime[w]) + Inv[t].get((f, c), 0) - x_prime
                    else:
                        x_prime = 0
                        Inv_prime = sum(y_prime_ftc[t].get((f, i, c), 0) for i in I_prime[w]) + Inv[t].get((f, c), 0)
                    x_bar_ct[c, t] += x_prime
                    Inv[t-1][(f, c)] = Inv_prime
                    w_bar_ft[f, t-1] += v_agg[c] * Inv_prime


                else:
                    x_prime = min(sum(y_prime_ftc[t].get((f, i, c), 0) for i in I_prime[w]) + Inv[t].get((f, c), 0), (total_c[c,t] - x_bar_ct[c, t]))
                    Inv_prime = 0
                    x_bar_ct[c, t] += x_prime

                x_prime_ft[t][(f, c)] = x_prime

            else:
                continue


    return y_prime_ftc, x_prime_ft, Inv

def solve_disagg_stage2(W,w_F,C,p_c,agg_params,agg_results,org_params,disagg_results_1):
    I, T, F,cap= org_params["I"], org_params["T"], org_params["F"], org_params["cap"]
    sh_w, h_w, tr_w,v_agg,demandp=agg_params["sh_w"], agg_params["h_w"], agg_params["tr_w"], agg_params["v_agg"],agg_params["demandp"]
    x_bar_w=agg_results['x_bar_w']     
    I_prime, z_bar_w_phi, new_demand_zone_numbers = (
        disagg_results_1["I_prime"],
        disagg_results_1["z_bar_w_phi"],
        disagg_results_1["new_demand_zone_numbers"])
    
    start_time = time.time()
    sorted_bundles_per_period = sort_bundles_stage2(W, C, T, I_prime, sh_w, h_w, tr_w, w_F, z_bar_w_phi, v_agg)
    cap1 = {(f, t): cap[f] for f in F for t in T}

    x1_values, y1_values, Inv_2_values = {}, {}, {}
    
    for w in W:
        y_prime_ftc, x_prime_ft, Inv = assign_bundles(w, T, C, z_bar_w_phi, x_bar_w, cap1, w_F, v_agg, I_prime, sh_w, h_w, tr_w, new_demand_zone_numbers,sorted_bundles_per_period)
        x1_values[w] = {(f, c, t): x_prime_ft[t].get((f,c),0) for t in T for c in C for f in w_F[w]}
        y1_values[w] = {(f, i,c, t): y_prime_ftc[t].get((f, i, c),0) for t in T for i in I_prime[w] for c in C for f in w_F[w]}
        Inv_2_values[w] = {(f, c,t): Inv[t].get((f, c),0) for t in T  for c in C for f in w_F[w] }
        

    end_time = time.time()
    print(f" Stage 2 Disaggregation Completed in {end_time - start_time:.2f} seconds")
    
    y_bar_c={}
    for c in C:
        y_bar_c[c] = {(f,i, t): y1_values[w].get((f, i, c, t),0) for w in W for f in w_F[w] for i in I_prime[w] for t in T}
    x_bar_c={}
    for c in C:
        x_bar_c[c]={(f,t): x1_values[w].get((f,c,t),0) for w in W for f in w_F[w] for t in T }

    inv_bar_c={}
    for c in C:
        inv_bar_c[c]={(f,t): Inv_2_values[w].get((f,c,t),0) for w in W for f in w_F[w] for t in T }    
    
    F_c = {c: [f for f in F if any(y_bar_c[c].get((f,i, t),0) > 0 for i in I for t in T)] for c in C}
	
    demand_zone_numbers = {}
    current_number = 1  
    for c in C:
        for p in p_c[c]:
            for i in I:
                if (i,p) not in demand_zone_numbers:
                    demand_zone_numbers[(i,p)] = current_number
                    current_number += 1
    
    demand_phi = {c:{} for c in C}
    for c in C:
        for p in p_c[c]:
                for i in I:
                    for t in T:
                        phi = (i,p)
                        new_phi = demand_zone_numbers.get(phi)
                        if new_phi is not None:
                            demand_value = demandp[c].get((i,p, t))
                            if demand_value is not None:
                                demand_phi[c][(new_phi, t)] = demand_value

    disagg_results_2 = {
        "x1_values": x1_values,
        "y1_values": y1_values,
        "Inv_2_values": Inv_2_values,
        "y_bar_c": y_bar_c,
        "x_bar_c":x_bar_c, 
        "inv_bar_c":inv_bar_c ,
        "F_c":F_c ,
        "demand_phi": demand_phi,
        "demand_zone_numbers": demand_zone_numbers,
    }


    return disagg_results_2


### Stage_3

def sort_bundles_stage3(C, T, I, p_c, sh_c, h_c, tr_c, F_c, demandp, v):
    sorted_bundles_per_period1 = {c: [] for c in C}
    
    for c in C:
        for t in T:
            sorted_bundles = sorted(
                [(i, p) for p in p_c[c] for i in I],
                key=lambda phi: (
                    sum(
                        sh_c[c][(f, phi[1], t)] +  
                        h_c[c][(f, phi[1], t)] +   
                        tr_c[c][(f, phi[0], phi[1])]  
                        for f in F_c[c]
                    ) * demandp[c].get((phi[0], phi[1], t), 0) * v[phi[1]]
                ),
                reverse=True
            )
            sorted_bundles_per_period1[c].append(sorted_bundles)
    
    return sorted_bundles_per_period1



def find_sorted_warehouses_for_bundlep(c, F_c, w_bar_ft, t, cap2, i, p, sh_c, h_c, tr_c, v):
    warehouse_costs = []

    for f in F_c[c]:
        if w_bar_ft[f, t] + v[p] <= cap2.get((f, t),0):
            if t >= 2:
                cost = (sh_c[c][(f, p, t)] + h_c[c][(f, p, t-1)] + tr_c[c][(f,i,p)])
                warehouse_costs.append((f, cost))
            else:
                cost = (sh_c[c][(f, p, t)] + h_c[c][(f, p, t)] + tr_c[c][(f,i,p)])
                warehouse_costs.append((f, cost))

    # Sort warehouses based on cost
    sorted_warehouses= sorted(warehouse_costs, key=lambda x: x[1])

    # Return only the list of warehouses
    return [f for f, _ in sorted_warehouses]


def assign_bundlesp(c, T,I, p_c, demand_phi, demand_zone_numbers, x_bar_c, inv_bar_c,y_bar_c, v_agg, 
                    F_c, sh_c, h_c, tr_c, v, sorted_bundles_per_period1):

    demand_phi_copy = demand_phi[c].copy()
    cap2 = {(f, t): (x_bar_c[c].get((f, t),0)+inv_bar_c[c].get((f,t-1),0)) *v_agg[c] for f in F_c[c] for t in T}
    total1=x_bar_c[c]
    y_bar_cc = {(f,i, t): y_bar_c[c].get((f,i, t),0) for f in F_c[c] for i in I for t in T}
    w_bar_ft = {(f, t): 0 for f in F_c[c] for t in T}
    w_bar1_ft = {(f,i, t): 0 for f in F_c[c] for i in I for t in T}
    x_bar_ft = {(f, t): 0 for f in F_c[c] for t in T}
    y2_prime_ftc = {t: {} for t in T}
    x2_prime_ft = {t: {} for t in T}
    Inv3 = {t: {} for f in F_c[c] for p in p_c[c] for t in T}
    fc = {t: [] for t in T}
    remove_duplicate = {t: [] for t in T}
    f_star = {}


    for t in reversed(T):
        y2_prime_ftc[t] = {}
        x2_prime_ft[t] = {}

        for (i,p) in sorted_bundles_per_period1[c][t-1]:
            phi = demand_zone_numbers[i,p]
            assigned = False
            offset_value = 0
            while demand_phi_copy[phi, t] > 0:

                warehouses = find_sorted_warehouses_for_bundlep(c, F_c, w_bar_ft, t, cap2, i, p, sh_c, h_c, tr_c, v)
                original_demand = demand_phi_copy[phi, t]
                wrapped_warehouses = warehouses[offset_value:offset_value+1] + warehouses[:offset_value]+warehouses[offset_value+1:]

                for f_star in wrapped_warehouses:
                    if y_bar_cc.get((f_star,i,t),0)-w_bar1_ft[f_star,i ,t]>0:
                        if w_bar_ft[f_star, t] + v[p] <= cap2.get((f_star, t),0):
                            y_prime =math.floor(min(demand_phi_copy[phi, t], ((cap2.get((f_star, t),0) - w_bar_ft[f_star, t])/v[p] ), (y_bar_cc.get((f_star,i,t),0)-w_bar1_ft[f_star,i, t])))
                            y2_prime_ftc[t][(f_star, i, p)] = y_prime
                            w_bar_ft[f_star, t] +=  v[p]*y_prime
                            demand_phi_copy[phi, t] -= y_prime
                            w_bar1_ft[f_star,i, t] += y_prime
                            if demand_phi_copy[phi, t] ==0:
                                assigned=True
                                break

                if assigned==True:

                    break

                elif assigned==False:
                    for (f_star, _, _) in list(y2_prime_ftc[t].keys()):
                        if (f_star, i, p) in y2_prime_ftc[t]:
                            y2_prime_value = y2_prime_ftc[t][(f_star, i, p)]
                            w_bar_ft[f_star, t] -= v[p] * y2_prime_value
                            w_bar1_ft[f_star,i, t]-=y2_prime_value
                            y2_prime_value=0

                    demand_phi_copy[phi, t] = original_demand

                    if offset_value >= len(warehouses):
                        break
                    else:
                        offset_value+=1


        remove_duplicate[t]=[]


        fc[t] = sorted(
            [(f, p) for (f, i, p), value in y2_prime_ftc[t].items()] +
            [(f, p) for (f, p), value in Inv3[t].items() if value > 0],
            key=lambda x: Inv3[t].get((x[0], x[1]), 0) + sum(y2_prime_ftc[t].get((x[0], i, x[1]),0) for i in I),
            reverse=True
        )

        for (f,p) in fc[t]:
            if (f,p) not in remove_duplicate[t]:
                remove_duplicate[t].append((f,p))

                if t==4:

                    if x_bar_ft[f, t]< total1.get((f, t),0):
                        x_prime = min((sum(y2_prime_ftc[t].get((f, i,p), 0) for i in I)), (total1.get((f, t),0) - x_bar_ft[f, t]) )
                        if x_prime==sum(y2_prime_ftc[t].get((f, i,p), 0) for i in I):
                            Inv_prime=0
                        else:
                            Inv_prime=sum(y2_prime_ftc[t].get((f,i,p), 0) for i in I) - x_prime

                    else:
                        x_prime=0
                        Inv_prime=sum(y2_prime_ftc[t].get((f,i, p), 0) for i in I)


                    x_bar_ft[f, t] += x_prime
                    Inv3[t-1][(f,p)]=Inv_prime
                    w_bar_ft[f,t-1] += v[p] * Inv_prime

                elif 1<t<4:

                    if  x_bar_ft[f, t]<total1.get((f, t),0) :
                        x_prime = min((sum(y2_prime_ftc[t].get((f,i,p), 0) for i in I)+Inv3[t].get((f, p), 0)), (total1.get((f, t),0) - x_bar_ft[f, t]))
                        if x_prime==sum(y2_prime_ftc[t].get((f, i,p), 0) for i in I)+Inv3[t].get((f, p), 0):
                            Inv_prime=0
                        else:
                            Inv_prime=sum(y2_prime_ftc[t].get((f, i,p), 0) for i in I)+Inv3[t].get((f, p), 0) - x_prime


                    else:
                        x_prime=0
                        Inv_prime=sum(y2_prime_ftc[t].get((f, i,p), 0) for i in I)+Inv3[t].get((f, p), 0)


                    x_bar_ft[f, t] += x_prime
                    Inv3[t-1][(f,p)]=Inv_prime
                    w_bar_ft[f,t-1] += v[p] * Inv_prime

                else:

                    x_prime = min(sum(y2_prime_ftc[t].get((f,i, p), 0) for i in I)+Inv3[t].get((f, p), 0),(total1.get((f, t),0) - x_bar_ft[f, t]))
                    Inv_prime=0


                    x_bar_ft[f, t] += x_prime
                x2_prime_ft[t][(f,p)] = x_prime

            else:
                continue



    return y2_prime_ftc,x2_prime_ft,Inv3


def solve_disagg_stage3(C, p_c,tr,org_params, agg_params,disagg_results_2):
   
    T,I,sh,h,v = org_params["T"], org_params["I"], org_params["sh"], org_params["h"], org_params["v"]
    sh_c, h_c, tr_c, demandp,v_agg=agg_params["sh_c"],agg_params["h_c"],agg_params["tr_c"], agg_params["demandp"],agg_params["v_agg"]
    demand_phi, demand_zone_numbers,F_c =disagg_results_2["demand_phi"],disagg_results_2["demand_zone_numbers"],disagg_results_2["F_c"]
    x_bar_c, inv_bar_c, y_bar_c=disagg_results_2["x_bar_c"],disagg_results_2["inv_bar_c"],disagg_results_2["y_bar_c"]

    start_time = time.time()

    sorted_bundles_per_period1 = sort_bundles_stage3(C, T, I, p_c, sh_c, h_c, tr_c, F_c, demandp, v)
    

    x2_values, y2_values, Inv_3_values = {}, {}, {}
    
    for c in C:

        y2_prime_ftc,x2_prime_ft,Inv3= assign_bundlesp(c, T,I, p_c, demand_phi, demand_zone_numbers, x_bar_c, inv_bar_c, y_bar_c,v_agg, F_c, sh_c, h_c, tr_c, v, sorted_bundles_per_period1)

        x2_values[c] = {(f, p, t): val for t in T for f in F_c[c] for p in p_c[c] if (val := x2_prime_ft[t].get((f, p), 0)) > 0}
        y2_values[c] = {(f, i, p, t): val for t in T for f in F_c[c] for p in p_c[c] for i in I if (val := y2_prime_ftc[t].get((f, i, p), 0)) > 0}
        Inv_3_values[c] = {(f, p, t): val for t in T for f in F_c[c] for p in p_c[c] if (val := Inv3[t].get((f, p), 0)) > 0}
    end_time = time.time()
    print("stage 3 Disaggregation Completed:", end_time - start_time, "seconds")    
    Greedy_obj=sum(sh[f,p,t]*x2_values[c].get((f,p,t),0) for c in C for f in F_c[c] for p in p_c[c]  for t in T)+sum(h[f,p,t]*Inv_3_values[c].get((f,p,t),0)  for c in C for f in F_c[c] for p in p_c[c]  for t in T)+sum(tr[f,i,p]*y2_values[c].get((f,i,p,t),0)for c in C for f in F_c[c] for i in I  for p in p_c[c]  for t in T)
    
    return Greedy_obj
