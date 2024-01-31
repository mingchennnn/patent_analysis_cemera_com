import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
import matplotlib.patches as mpatches

def gini_coefficient(data):
    """Calculate the Gini coefficient of a numpy array."""
    # Filter out 'NaN' values
    data = np.array(data)[~np.isnan(data)]
    # All values are sorted and normalized (made non-negative).
    data = np.sort(data)
    n = data.size
    index = np.arange(1, n+1)
    # The Gini coefficient is calculated as per formula.
    return (np.sum((2 * index - n - 1) * data)) / (n * np.sum(data))

def theil_index(data):
    """Calculate the Theil index of a numpy array."""
    # Filter out 'NaN' values
    data = np.array(data)[~np.isnan(data)]
    mean_data = np.mean(data)
    # Ensuring no zero values to avoid division errors.
    data = np.where(data == 0, np.nan, data)
    # Theil index calculation.
    T = np.nansum((data / mean_data) * np.log(data / mean_data)) / len(data)
    return T

def hoover_index(data):
    """Calculate the Hoover index of a numpy array."""
    # Filter out 'NaN' values
    data = np.array(data)[~np.isnan(data)]
    mean_data = np.mean(data)
    # The sum of absolute differences between each data point and the mean, divided by twice the total sum.
    H = np.sum(np.abs(data - mean_data)) / (2 * np.sum(data))
    return H

def create_threshold_graph(G, edge_threshold):
    # Create a new graph H with the same nodes as G
    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))

    # Iterate over the edges of G and add only those edges to H that meet the threshold criteria
    for u, v, data in G.edges(data=True):
        if data.get('weight', 0) > edge_threshold:  # Check if edge weight is above the threshold
            H.add_edge(u, v)  # Add the edge to H with all its attributes

    return H

def weighted_degree_centrality(G):
    # Initialize a dictionary to store weighted degree centrality
    centrality = {}
    
    # Iterate over all nodes in the graph
    for node in G.nodes():
        # Sum the weights of the edges incident to the node
        weighted_degree = sum(weight for _, _, weight in G.edges(node, data='weight'))
        # Add the node and its weighted degree to the centrality dictionary
        centrality[node] = weighted_degree

    # The maximum possible sum of weighted degree is just N-1 since the maximum weight is 1
    max_weighted_degree = len(G.nodes)-1
    
    # Normalize the centralities
    for node in centrality:
        centrality[node] /= max_weighted_degree
    
    return centrality

def weighted_closeness_centrality(G):
    # Copy the graph to avoid modifying the original graph
    H = G.copy()

    # Invert the weights in the copied graph
    for u, v, data in H.edges(data=True):
        # Ensure the weight is non-zero to avoid division by zero
        if data['weight'] != 0:
            data['inv_weight'] = 1.0 / data['weight']
        else:
            data['inv_weight'] = float('inf')

    # Compute the closeness centrality using the inverted weights
    closeness = nx.closeness_centrality(H, distance='inv_weight')
    return closeness

def weighted_betweenness_centrality(G, normalized=True):
    # Copy the graph to avoid modifying the original graph
    H = G.copy()

    # Invert the weights in the copied graph
    for u, v, data in H.edges(data=True):
        # Ensure the weight is non-zero to avoid division by zero
        if data['weight'] != 0:
            data['inv_weight'] = 1.0 / data['weight']
        else:
            # Assign a very large weight to edges with zero weight to simulate 'infinite' distance
            data['inv_weight'] = float('inf')

    # Compute the betweenness centrality using the inverted weights
    betweenness = nx.betweenness_centrality(H, normalized=normalized, weight='inv_weight')
    return betweenness

def weighted_eigenvector_centrality(G):
    centrality = nx.eigenvector_centrality_numpy(G, weight='weight')
    return centrality

def weighted_harmonic_centrality(G):
    # Initialize a dictionary to hold the harmonic centrality values
    centrality = {}
    
    # Iterate over all nodes in the graph
    for node in G.nodes():
        path_lengths = nx.single_source_dijkstra_path_length(G, node, weight='weight')
        
        # Sum the reciprocal of the path lengths (excluding the node itself)
        centrality[node] = sum(1 / length for target, length in path_lengths.items() if target != node and length > 0)
        
    return centrality

def weighted_katz_centrality(G, alpha=0.005, beta=1.0):
    """
    Calculate Katz centrality for a weighted graph G.
    
    Parameters:
    - G: NetworkX graph
    - alpha: Attenuation factor (default: 0.005)
    - beta: Weight scalar for each node (default: 1.0, can be a scalar or a dictionary)
    
    Returns:
    - Dictionary of nodes with Katz centrality as the value
    """
    
    # Calculate Katz centrality considering the edge weights
    centrality = nx.katz_centrality(G, alpha=alpha, beta=beta, normalized=True, weight='weight')
    
    return centrality

def weighted_pagerank_centrality(G, alpha=0.85):
    """
    Calculate PageRank centrality for a weighted graph G.
    
    Parameters:
    - G: NetworkX graph
    - alpha: Damping parameter for PageRank, representing the probability
             at each step that the person will continue clicking on links,
             as opposed to stopping (default: 0.85).
    
    Returns:
    - Dictionary of nodes with PageRank centrality as the value
    """
    
    # Calculate PageRank considering the edge weights
    pagerank = nx.pagerank(G, alpha=alpha, weight='weight')
    
    return pagerank

def weighted_clustering_coeff(G):
    clustering = nx.clustering(G, weight='weight')
    return clustering

def vertex_connectivity(G):
    # Check if the graph is connected, as vertex connectivity is defined for connected graphs
    if nx.is_connected(G):
        # Calculate the vertex connectivity
        connectivity = nx.node_connectivity(G)
    else:
        # If the graph is not connected, its vertex connectivity is 0
        connectivity = 0
    print(connectivity)

    return {'value': connectivity}

def weighted_vertex_connectivity(G):
    # Step 1: Calculate vertex strength based on edge weights
    vertex_strength = {v: sum(data['weight'] for u, v, data in G.edges(v, data=True)) for v in G.nodes}

    # Step 2: Sort vertices by strength in descending order
    sorted_vertices = sorted(vertex_strength, key=vertex_strength.get, reverse=True)

    # Step 3: Iteratively remove vertices and check connectivity
    for i in range(1, len(sorted_vertices) + 1):
        print(i, len(sorted_vertices))
        for vertices_to_remove in combinations(sorted_vertices, i):
            H = G.copy()
            H.remove_nodes_from(vertices_to_remove)
            if not nx.is_connected(H):
                print('result', i)
                return {'value': i}  # Return the size of the minimum set of vertices whose removal disconnects G

    # If the graph remains connected regardless, return the total number of vertices
    return {'value': len(G.nodes)}



def sum_edge_weights(G):
    total_weight = sum(weight for _, _, weight in G.edges(data='weight'))
    return {'value': total_weight}

def ave_edge_weight(G):
    total_weight = sum_edge_weights(G)['value']
    average_weight = total_weight / G.number_of_nodes()
    return {'value':average_weight}

def ave_edge_weight_per_patent(G, patent_number_year):
    total_weight = sum_edge_weights(G)['value']
    average_weight = total_weight / patent_number_year
    return {'value':average_weight}

def ave_shortest_path_length(G):
    # Copy the graph to avoid modifying the original graph
    H = G.copy()

    # Invert the weights in the copied graph
    for u, v, data in H.edges(data=True):
        # Ensure the weight is non-zero to avoid division by zero
        if data['weight'] != 0:
            data['inv_weight'] = 1.0 / data['weight']
        else:
            # Assign a very large weight to edges with zero weight to simulate 'infinite' distance
            data['inv_weight'] = float('inf')
    ave_shortest_path_length = nx.average_shortest_path_length(H, weight='inv_weight')
    return {'value': ave_shortest_path_length}

def ave_clustering_coeff(G):
    return {'value': nx.average_clustering(G, weight='weight')}

def weighted_small_worldness(G, niter=100, nrand=10):
    """
    Calculate the weighted small-worldness of graph G.

    Parameters:
    - G: NetworkX graph (weighted)
    - niter, nrand: Parameters for the approximate average shortest path length calculation

    Returns:
    - sigma: Weighted small-worldness of the graph
    """
    # Calculate the weighted clustering coefficient of G
    C = nx.average_clustering(G, weight='weight')

    # Calculate the average shortest path length of G using Dijkstra's algorithm
    L = nx.average_shortest_path_length(G, weight='weight')

    # Generate a random graph with the same number of nodes, edges, and weight distribution as G
    rand_G = nx.gnm_random_graph(G.number_of_nodes(), G.number_of_edges())
    for u, v, data in rand_G.edges(data=True):
        # Randomly assign weights from the original graph to the random graph
        data['weight'] = np.random.choice([d['weight'] for x, y, d in G.edges(data=True)])

    # Calculate the weighted clustering coefficient and average shortest path length of the random graph
    C_rand = nx.average_clustering(rand_G, weight='weight')
    L_rand = nx.average_shortest_path_length(rand_G, weight='weight')

    # Calculate weighted small-worldness
    sigma = (C / C_rand) / (L / L_rand)

    return {'value': sigma}

def weighted_degree_assortativity(G, weight='weight'):
    # Calculate weighted degree (strength) for each node
    strengths = dict(G.degree(weight=weight))

    # Create a new graph to represent the weighted degree as node attributes
    WG = nx.Graph()
    for u, v, data in G.edges(data=True):
        # Add edges between nodes with weights equal to the product of node strengths
        WG.add_edge(u, v, weight=(strengths[u] * strengths[v]))

    # Calculate assortativity of the new graph, treating the edge weights as connection strengths
    assortativity = nx.degree_assortativity_coefficient(WG, weight='weight')
    return {'value': assortativity}

# Get the size of company by their patent counts
def get_patent_number(start_year, end_year):
    # Load the CSV file into a DataFrame
    df = pd.read_csv("./output/mapped_company_id_year_tokens.csv")

    # Group by 'company' and 'year', then count the number of rows for each group
    patent_counts = df.groupby(['company', 'year']).size().reset_index(name='number_patents')

    # Create a DataFrame of all combinations of companies and years from 1970 to 2022
    companies = df['company'].unique()
    years = range(start_year, end_year+1)
    all_combinations = pd.MultiIndex.from_product([companies, years], names=['company', 'year']).to_frame(index=False)

    # Merge the complete combinations with the patent counts
    complete_patent_counts = all_combinations.merge(patent_counts, on=['company', 'year'], how='left')

    # Replace NaN values with 0 in 'number_patents' column
    complete_patent_counts['number_patents'] = complete_patent_counts['number_patents'].fillna(0)

    # Convert 'number_patents' to integer (as it may be float due to NaN filling)
    complete_patent_counts['number_patents'] = complete_patent_counts['number_patents'].astype(int)
    return complete_patent_counts

# Function to generate a measure of innovation
def network_measure_innovation(start_year, end_year, measure_function, patent_count_data, edge_threshold):
    measure_by_year = {}
    top_companies_by_year = []
    for year in range(start_year, end_year+1):
        # Load the cosine similarity matrix
        cos_sim_df = pd.read_csv(f"./output/cosine_similarity/cosine_similarity_matrix_{year}.csv")

        # Filter patent counts for the current year
        year_patent_counts = patent_count_data[patent_count_data['year'] == year]
        
        # Create a graph
        G = nx.Graph()


        # Add nodes with sizes based on log of patent counts
        for company in cos_sim_df.columns[1:]:
            patent_count = year_patent_counts[year_patent_counts['company'] == company]['number_patents'].values[0]
            if patent_count > 0:
                G.add_node(company)
        # Add edges with weights based on cosine similarity
        for i, company_i in enumerate(cos_sim_df.columns[1:]):
            for j, company_j in enumerate(cos_sim_df.columns[1:]):
                if i < j:
                    sim = cos_sim_df.iloc[i, j + 1]
                    if company_i in G.nodes and company_j in G.nodes:
                        G.add_edge(company_i, company_j, weight=sim)
        
        # Prepare the attribute for the functions to generate results
        patent_number_year = patent_count_data.groupby('year')['number_patents'].sum()[year]
        # G_threshold = create_threshold_graph(G, edge_threshold)
        # Use the pre-defined function to get the result
        if measure_function == ave_edge_weight_per_patent:
            measure = measure_function(G, patent_number_year)
        # elif measure_function == vertex_connectivity:
        #     measure = measure_function(G_threshold)
        else:
            measure = measure_function(G)
        measure_values = list(measure.values())  # Get centrality values for this year
        measure_by_year[year] = measure_values  # Store values
        # Get sorted measures
        sorted_measure = [item[0] for item in sorted(measure.items(), key=lambda x: x[1], reverse=True)]
        # Only append for every 2 years
        if year % 2 == 0:
            top_companies_by_year.append(sorted_measure[:5])
        print(year)
    return measure_by_year, top_companies_by_year

def statistic_measure_innovation(start_year, end_year, measure_function, patent_count_data):
    patent_number = list(patent_count_data.groupby('year')['number_patents'].sum())
    # Aggregate the count of unique companies per year
    patent_count_data = patent_count_data[patent_count_data['number_patents'] > 0].groupby('year')['company'].nunique().reset_index()
    firm_number = patent_count_data['company'].tolist()
    ave_patent_number = [i/j for i,j in zip(patent_number, firm_number)]
    if measure_function == 0:
        return patent_number
    elif measure_function == 1:
        return firm_number
    elif measure_function == 2:
        return ave_patent_number

def generate_plots(measure_function):
    # If statistics without networks
    if measure_function[3]==0:
        plt.figure(figsize=(20, 10))
        # Aggregate 'number_patents' for each year across all companies
        measure = statistic_measure_innovation(start_year, end_year, measure_function[0], get_patent_number(start_year, end_year))
        plt.plot(years, measure)
        plt.ylabel(f'{measure_function[1]}')
        plt.title(f'{measure_function[1]} by Year')
        plt.ylim(bottom=0)
        
    # If a boxplot is needed for distribution
    elif measure_function[3]==1:
        fig, ax1 = plt.subplots(2, 1, figsize=(20, 12), gridspec_kw={'height_ratios': [4, 1]})
        measure_result, top_companies = network_measure_innovation(start_year, end_year, measure_function[0], get_patent_number(start_year, end_year), edge_threshold)
        df_measure = pd.DataFrame.from_dict(measure_result, orient='index')
        df_measure = df_measure.transpose()  # Transpose to have years as columns
        # Plotting the boxplot
        ax1[0].boxplot([np.array(df_measure[year])[~np.isnan(df_measure[year])] for year in years], positions=years,
                    patch_artist=True, boxprops=dict(color='C0',facecolor='none'), medianprops = dict(color='C0'))
        ax1[0].set_ylabel(f'{measure_function[1]}', color='C0')
        ax1[0].tick_params(axis='y', labelcolor='C0')
        ax1[0].set_title(f'Distribution of {measure_function[1]} Across Companies by Year')
        proxy_artist1 = mpatches.Patch(color='C0', label='Boxplot of Centrality')
        # Plotting gini coefficient or other inequality measures
        gini = [gini_coefficient(list(df_measure[year])) for year in years]
        # gini = [theil_index(list(df_measure[year])) for year in years]
        # gini = [hoover_index(list(df_measure[year])) for year in years]
        ax2 = ax1[0].twinx()
        ax2.set_ylabel('Gini Coefficient', color='C1')
        ax2.tick_params(axis='y', labelcolor='C1')
        # ax2.set_ylim(0, 1)
        line2, = ax2.plot(years, gini, label='Gini Coefficient', color='C1')
        ax1[0].legend(handles=[proxy_artist1, line2])
        # Create a table for top 5 companies
        ax1[1].axis('off')
        top_companies = np.array(top_companies).transpose()
        table = plt.table(cellText=top_companies,
                          colLabels=[year for year in years if year % 2 == 0],
                          rowLabels=['Top1', 'Top2', 'Top3', 'Top4', 'Top5'],
                          loc='center',
                          cellLoc='center', colLoc='center',
                          bbox=[0, -0.5, 1, 0.3])
        # Manually set font size for the table
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        
    # If a single value is computed
    else:
        plt.figure(figsize=(20, 10))
        measure, _ = network_measure_innovation(start_year, end_year, measure_function[0], get_patent_number(start_year, end_year), edge_threshold)
        plt.plot(years, [item[0] for item in measure.values()])
        plt.ylabel(f'{measure_function[1]}')
        plt.title(f'{measure_function[1]} by Year')
        if measure_function[0] != weighted_degree_assortativity:
            plt.ylim(bottom=0)

    plt.xticks([year for year in years if year % 5 == 0], [year for year in years if year % 5 == 0])
    plt.xlabel('Year')

    plt.savefig(f'./output/network_measures/{measure_function[2]}.png')

# Parameters for graph drawing
start_year = 1970
end_year = 2022
edge_threshold = 0.3    # set threshold for edge visibility
seed = 5    # find the seed that generate the best graph

# Create a list of DataFrame from the centrality data, if measure_function[2] == 0, it's a single measure without building a network;
# if == 1, it's a distribution 
measure_functions = []
measure_functions.append([0, 'Total Patent Number', 'total_patent_number', 0])
measure_functions.append([1, 'Firm Number', 'firm_number', 0])
measure_functions.append([2, 'Average Patent Number', 'ave_patent_number', 0])
measure_functions.append([weighted_degree_centrality, 'Degree Centrality', 'degree_centrality', 1])
measure_functions.append([weighted_closeness_centrality, 'Closeness Centrality', 'closeness_centrality', 1])
measure_functions.append([weighted_betweenness_centrality, 'Betweenness Centrality', 'betweenness_centrality', 1])
measure_functions.append([weighted_eigenvector_centrality, 'Eigenvalue Centrality', 'eigenvector_centrality', 1])
measure_functions.append([weighted_harmonic_centrality, 'Harmonic Centrality', 'harmonic_centrality', 1])
measure_functions.append([weighted_katz_centrality, 'Katz Centrality', 'katz_centrality', 1])
measure_functions.append([weighted_pagerank_centrality, 'PageRank Centrality', 'pagerank_centrality', 1])
measure_functions.append([weighted_clustering_coeff, 'Clustering Coefficient', 'clustering_coeff', 1])
measure_functions.append([sum_edge_weights, 'Sum of Degree', 'sum_degree', 2])
measure_functions.append([ave_edge_weight, 'Average Degree', 'ave_degree', 2])
measure_functions.append([ave_edge_weight_per_patent, 'Average Degree per Patent', 'ave_degree_per_patent', 2])
measure_functions.append([ave_shortest_path_length, 'Average Shortest Path Length', 'ave_shortest_path_length', 2])
measure_functions.append([ave_clustering_coeff, 'Average Clustering Coefficient', 'ave_clustering_coeff', 2])
measure_functions.append([weighted_small_worldness, 'Small-World Coefficient', 'small_worldness', 2])
# measure_functions.append([vertex_connectivity, 'Node Connectivity', 'node_connectivity', 2])
# measure_functions.append([weighted_vertex_connectivity, 'Node Connectivity', 'node_connectivity', 2])


# Prepare the plot
years = range(start_year, end_year+1)
measure_function = [weighted_degree_assortativity, 'Degree Assortativity', 'degree_assortativity', 2]
generate_plots(measure_function)

'''
for measure_function in measure_functions:
    generate_plots(measure_function)
    print(measure_function[1])
'''


