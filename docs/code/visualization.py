import matplotlib.pyplot as plt

def plot_region(org_params):
    df,dp,F=org_params['df'], org_params['dp'], org_params['F']
    # Extract demand point coordinates and population ratio
    dp_lats = dp['INTPTLAT']
    dp_longs = dp['INTPTLONG']
    dp_sizes = dp['pop_ratio'] * 5000  # Scale for better visualization
    
    # Extract warehouse coordinates
    warehouse_data = df[df['Code'].isin(F)]  # Select only the sampled warehouses
    wh_lats = df['Latitude']
    wh_longs = df['Longitude']
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(dp_longs, dp_lats, s=dp_sizes, c='blue', alpha=0.6, edgecolors='black', label='Demand Points (Scaled by Population)')
    ax.scatter(wh_longs, wh_lats, s=100, c='red', marker='^', label='Warehouses')
    
    # Labels and aesthetics
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Demand Points and Warehouses Visualization")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Show the plot
    plt.show()



def plot_clusters(dp_reduced, df_reduced):
 
    
    fig, ax = plt.subplots(figsize=(12, 8))

    scatter_dp = ax.scatter(dp_reduced['INTPTLONG'], dp_reduced['INTPTLAT'], 
                            s=dp_reduced['pop_ratio'] * 5000, c=dp_reduced['cluster_label'], 
                            cmap='tab10', alpha=0.85, edgecolors='black', label="Demand Clusters")

  
    scatter_df = ax.scatter(df_reduced['Longitude'], df_reduced['Latitude'], 
                            s=150, c=df_reduced['cluster_label'], 
                            cmap='tab10', marker='^', edgecolors='black', label="Warehouse Clusters")


    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Clustered Demand Points and Warehouses")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.subplots_adjust(bottom=0.25)  

    cbar_ax1 = fig.add_axes([0.15, 0.1, 0.7, 0.02])  
    cbar_ax2 = fig.add_axes([0.15, 0.05, 0.7, 0.02])  

    cbar_dp = plt.colorbar(scatter_dp, cax=cbar_ax1, orientation='horizontal', label="Demand Cluster ID")
    cbar_df = plt.colorbar(scatter_df, cax=cbar_ax2, orientation='horizontal', label="Warehouse Cluster ID")


    plt.show()





def plot_results(results_df):
    clusters = results_df[('nf','nd','n_clusters')].apply(lambda x: f"({x[0]}, {x[1]}, {x[2]})")
    opt_gaps = results_df["opt_gap"]
    computation_times = results_df["total_time"]
    optimal_computation_time = results_df['time_opt']
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    ax1.set_xlabel("Cluster Composition (n_f, n_d, n_p)",fontsize=12, fontweight='bold')
    ax1.set_ylabel("Computation Time (seconds)", color='tab:blue', fontsize=12, fontweight='bold')
    ax1.plot(clusters, computation_times, marker='o', linestyle='-', color='tab:blue', label="Algorithm Computation Time")
    ax1.axhline(y=results_df['time_opt'].iloc[0], color='tab:gray', linestyle='dashed', label="Optimal Computation Time")  # Optimal time reference line
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, linestyle='--', alpha=0.6)
    

    ax2 = ax1.twinx()
    ax2.set_ylabel("Optimality Gap (%)", color='tab:red', fontsize=12, fontweight='bold')
    ax2.plot(clusters, opt_gaps, marker='s', linestyle='-', color='tab:red', label="Optimality Gap")
    ax2.tick_params(axis='y', labelcolor='tab:red')
    

    fig.suptitle("Optimality Gap and Computation Time for Different Cluster Compositions", fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    fig.tight_layout()
    
    
    ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax2.legend(loc='lower left', bbox_to_anchor=(1.02, 0), borderaxespad=0)
    
    
    plt.show()