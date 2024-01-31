import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import community as community_louvain
import matplotlib


def interpolate_positions(old_positions, new_positions, stepsize_ratio):
    interpolated_positions = {}

    for node in old_positions:
        if node in new_positions:  # Check if node exists in new_positions
            old_pos = np.array(old_positions[node])
            new_pos = np.array(new_positions[node])

            # Calculate the vector from the old position to the new position
            movement_vector = new_pos - old_pos

            # Apply the stepsize ratio to the movement vector
            interpolated_pos = old_pos + movement_vector * stepsize_ratio

            # Assign the interpolated position
            interpolated_positions[node] = tuple(interpolated_pos)
        else:
            # If the node does not exist in new_positions, use the old position
            interpolated_positions[node] = old_positions[node]

    # Handle nodes that are in new_positions but not in old_positions
    for node in new_positions:
        if node not in old_positions:
            interpolated_positions[node] = new_positions[node]

    return interpolated_positions


# Parameters for graph drawing
end_year = 2022
nodesize_multiplier = 30    # node size multiplier
edgeweight_amplifier = 4    # exaggeratio factor of edge weight 
edge_threshold = 0.3    # set threshold for edge visibility
fontsize_factor = 4    # set fontsize factor
stepsize_ratio = 1    # ratio of interpolation from old position to new postion
seed = 5    # find the seed that generate the best graph
plot_range = 1.2    # plot range
k_factor = 3    # the larger the more distant between nodes connected with edges

# Get the size of company by their patent counts
# Load the CSV file into a DataFrame
df = pd.read_csv("./output/mapped_company_id_year_tokens.csv")

# Group by 'company' and 'year', then count the number of rows for each group
patent_counts = df.groupby(['company', 'year']).size().reset_index(name='number_patents')

# Create a DataFrame of all combinations of companies and years from 1970 to 2022
companies = df['company'].unique()
years = range(1970, 2023)
all_combinations = pd.MultiIndex.from_product([companies, years], names=['company', 'year']).to_frame(index=False)

# Merge the complete combinations with the patent counts
complete_patent_counts = all_combinations.merge(patent_counts, on=['company', 'year'], how='left')

# Replace NaN values with 0 in 'number_patents' column
complete_patent_counts['number_patents'] = complete_patent_counts['number_patents'].fillna(0)

# Convert 'number_patents' to integer (as it may be float due to NaN filling)
complete_patent_counts['number_patents'] = complete_patent_counts['number_patents'].astype(int)

for year in range(1970, end_year+1):
    # Load the cosine similarity matrix
    cos_sim_df = pd.read_csv(f"./output/cosine_similarity/cosine_similarity_matrix_{year}.csv")

    # Filter patent counts for the current year
    year_patent_counts = complete_patent_counts[complete_patent_counts['year'] == year]
    
    # Create a graph
    G = nx.Graph()

    # Dictionary to store node sizes
    node_sizes = {}

    # Add nodes with sizes based on log of patent counts
    for company in cos_sim_df.columns[1:]:
        patent_count = year_patent_counts[year_patent_counts['company'] == company]['number_patents'].values[0]
        if patent_count > 0:
            node_size = patent_count*nodesize_multiplier  # Adjust multiplier as needed
            G.add_node(company)
            node_sizes[company] = node_size

    # Add edges with weights based on cosine similarity
    for i, company_i in enumerate(cos_sim_df.columns[1:]):
        for j, company_j in enumerate(cos_sim_df.columns[1:]):
            if i < j:
                sim = cos_sim_df.iloc[i, j + 1]
                if company_i in node_sizes and company_j in node_sizes:
                    alpha_value = 0.5 if sim > edge_threshold else 0  # Set alpha to 0.5 if sim > 0.1, else 0
                    G.add_edge(company_i, company_j, weight=sim, alpha=alpha_value)

    # Detect communities considering edge weights using the Louvain method
    partition = community_louvain.best_partition(G, weight='weight')

    # Draw the graph
    plt.figure(figsize=(50, 50))
    # Get position by the interpolation between new position and the position last year
    N_node = len(G.nodes)
    print(N_node)
    new_pos = nx.spring_layout(G, k=k_factor*N_node/69, seed=seed)    # or k=1/(N_node**(1/k_factor)
    if year == 1970:
        pos = new_pos
    else:
        pos = interpolate_positions(old_pos, new_pos, stepsize_ratio)

    # Get the colormap for different patitions
    cmap = matplotlib.colormaps.get_cmap('viridis')
    # Normalize the partition values for color mapping
    norm = matplotlib.colors.Normalize(vmin=0, vmax=max(partition.values()))
    node_colors = [cmap(norm(value)) for value in partition.values()]

    # Draw nodes with sizes. Only draw nodes that have a size attribute (i.e., patent_count > 0)
    nx.draw_networkx_nodes(G, pos, nodelist=node_sizes.keys(), node_size=list(node_sizes.values()), node_color=node_colors, alpha=0.5)
    
    # Draw edges with varying transparency
    for u, v, attrs in G.edges(data=True):
        edge_alpha = attrs['alpha']
        edge_weight = np.exp(attrs['weight']*edgeweight_amplifier)-1  # exaggerate the difference between of edges
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=edge_weight, alpha=edge_alpha)

    # Draw labels with font size proportional to patent count
    for company in G.nodes():
        patent_count = year_patent_counts[year_patent_counts['company'] == company]['number_patents'].values[0]
        if patent_count > 0:
            # Calculate font size, adjust scaling factor as needed
            font_size = max(20, 2*np.log(patent_count + 1) * fontsize_factor)  # Ensure a minimum font size
            label_pos = {company: pos[company]}  # Position for the current label
            nx.draw_networkx_labels(G, label_pos, labels={company: company}, font_size=font_size)

    # Add text denoting the number of partitions
    num_partitions = len(set(partition.values()))
    plt.text(0.05, 0.95, f'#Clusters: {num_partitions}', transform=plt.gca().transAxes, 
             horizontalalignment='left', verticalalignment='top', fontsize=50, bbox=dict(facecolor='white', alpha=0.8))

    plt.title(f"Company Cosine Similarity Network {year}", fontsize=50)
    plt.axis('off')
    plt.xlim(-plot_range, plot_range)
    plt.ylim(-plot_range, plot_range)
    plt.savefig(f"./output/network_graphs_clusters/cosine_similarity_{year}.png")
    # plt.savefig(f"./output/network_graphs/cosine_similarity_{year}.png")
    plt.close()
    # Preserve the old position to the next iteration
    old_pos = pos.copy()
    print(year)
    # plt.show()
