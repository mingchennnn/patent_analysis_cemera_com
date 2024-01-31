import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import pandas as pd
import numpy as np


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
nodesize_multiplier = 7    # node size multiplier
edgeweight_amplifier = 3    # exaggeration factor of edge weight 
edge_threshold = 0.3    # set threshold for edge visibility
fontsize_factor = 3    # fontsize factor
stepsize_ratio = 0.2    # ratio of interpolation from old position to new postion
seed = 5    # find the seed that generate the best graph
plot_range = 1    # plot range
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


# Pre-calculate positions for all companies
G_full = nx.Graph()
G_full.add_nodes_from(companies)
# pos = nx.spring_layout(G_full)  # Fixed positions based on the full set


# Function to create a graph for a specific year
def create_year_graph(year):
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

    return G, node_sizes

# Create a figure for the animation
fig, ax = plt.subplots(figsize=(20, 20))

# Animation update function
def update(year):
    global old_pos  # Declare old_pos as global to modify it
    ax.clear()
    G_year, node_sizes = create_year_graph(year)

    # Get position by the interpolation between new position and the position last year
    N_node = len(G_year.nodes)
    new_pos = nx.spring_layout(G_year, k=k_factor*N_node/69, seed=seed)    # or k=1/(N_node**(1/k_factor))
    if year == 1970 or not old_pos:
        pos = new_pos
    else:
        pos = interpolate_positions(old_pos, new_pos, stepsize_ratio)

    # Draw nodes with sizes. Only draw nodes that have a size attribute (i.e., patent_count > 0)
    nx.draw_networkx_nodes(G_year, pos, nodelist=node_sizes.keys(), node_size=list(node_sizes.values()), node_color='skyblue')
    
    # Draw edges with varying transparency
    for u, v, attrs in G_year.edges(data=True):
        edge_alpha = attrs['alpha']
        edge_weight = np.exp(attrs['weight']*edgeweight_amplifier)-1  # exaggerate the difference between of edges
        nx.draw_networkx_edges(G_year, pos, edgelist=[(u, v)], width=edge_weight, alpha=edge_alpha)


    # Draw labels with font size proportional to patent count
    for company in G_year.nodes():
        patent_count = complete_patent_counts[(complete_patent_counts['company'] == company) & 
                                              (complete_patent_counts['year'] == year)]['number_patents'].values[0]
        if patent_count > 0:
            font_size = max(10, np.log(patent_count + 1) * fontsize_factor)  # Adjust the scaling factor as needed
            label_pos = {company: pos[company]}  # Position for the current label
            nx.draw_networkx_labels(G_year, label_pos, labels={company: company}, font_size=font_size)

    ax.set_title(f"Company Cosine Similarity Network {year}", fontsize=30)
    plt.xlim(-plot_range, plot_range)
    plt.ylim(-plot_range, plot_range)
    ax.axis('off')
    # Preserve the old position to the next iteration
    old_pos = pos.copy()  # Use copy to avoid reference issues
    print(year)

# Create animation
ani = animation.FuncAnimation(fig, update, frames=range(1970, end_year+1), repeat=True)

# Save and show the animation
ani.save(f'./output/network_graphs/network_animation_1970{end_year}.gif', writer='pillow')
# plt.show()

