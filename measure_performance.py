import time
import os
import dimod
import numpy as np
import matplotlib.pyplot as plt
from quanfluence_sdk import QuanfluenceClient # Assuming this SDK is installed

# --- Quanfluence Client Setup ---
# These credentials and device_id are preconfigured as per the documentation
# IMPORTANT: Replace with your actual credentials or ensure they are configured
# in a way that QuanfluenceClient() can access them.
try:
    client = QuanfluenceClient()
    # Ensure you are signed in. If your client handles sign-in internally
    # or through environment variables, this explicit call might be adjusted.
    client.signin('rochester_user0','Rochesteruser@123') # Example credentials
except Exception as e:
    print(f"Error initializing Quanfluence Client: {e}")
    print("Please ensure the Quanfluence SDK is installed and configured correctly.")
    # Exit if client cannot be initialized, as the rest of the script depends on it.
    exit()

device_id = 16 # As specified in your script

# --- Function to generate a Max-Cut QUBO for a complete graph ---
def create_max_cut_qubo(num_nodes):
    """
    Generates a QUBO dictionary for the Max-Cut problem on a complete graph
    with 'num_nodes'.
    Edges have unit weights.
    """
    Q = {}
    # For a complete graph, add edges between all distinct pairs of nodes
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            weight = 1.0 # Assuming unit weights for a simple Max-Cut

            # QUBO formulation for Max-Cut:
            # For an edge (i,j) with weight 'w':
            # Maximize sum_edges w_ij (1 - x_i x_j) / 2  where x_i in {-1, 1}
            # Convert to binary x_i in {0, 1} using x_i' = (x_i + 1)/2
            # Leads to: Q_ii = sum_j w_ij, Q_jj = sum_i w_ij, Q_ij = -2 * w_ij
            # For the common QUBO form min y^T Q y, we want to maximize cuts.
            # A common formulation to maximize cuts (number of edges between sets):
            # Sum_{i<j} w_ij (x_i(1-x_j) + x_j(1-x_i))
            # = Sum_{i<j} w_ij (x_i + x_j - 2x_i x_j)
            # This means:
            # Linear terms (diagonal in QUBO): Q[(i,i)] -= w_ij
            # Quadratic terms (off-diagonal in QUBO): Q[(i,j)] += 2 * w_ij
            # (Note: Your original formulation seems to be a standard one, so sticking to it)

            Q[(i, j)] = Q.get((i, j), 0.0) + (2 * weight)
            Q[(i, i)] = Q.get((i, i), 0.0) - weight
            Q[(j, j)] = Q.get((j, j), 0.0) - weight
    
    # Ensure all values are 32-bit floating point numbers as required by Quanfluence
    for key in Q:
        Q[key] = np.float32(Q[key])

    return Q

# --- Function to generate and save the performance plot ---
def generate_and_save_plot(nodes_data, avg_times_data, filename="performance_plot.png"):
    """
    Generates a scatter plot of average execution times vs. number of nodes
    and saves it to a file.
    """
    plt.figure(figsize=(12, 7)) # Figure size
    plt.scatter(nodes_data, avg_times_data, label='Measured Average Time', color='blue', zorder=5, s=50) # Scatter plot

    # Font sizes for readability
    fontsize_title = 18
    fontsize_labels = 14
    fontsize_legend = 12
    fontsize_ticks = 12

    # Title and labels
    plt.title('Average Execution Time vs. Number of Nodes on Quanfluence Ising Machine', fontsize=fontsize_title)
    plt.xlabel('Number of Nodes', fontsize=fontsize_labels)
    plt.ylabel('Average Execution Time (seconds)', fontsize=fontsize_labels)

    # Grid and legend
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=fontsize_legend)

    # Tick parameters
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)

    # Layout adjustment
    plt.tight_layout()

    # Save the plot to the specified filename
    try:
        plt.savefig(filename)
        print(f"\nPlot successfully saved to {filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    
    # plt.show() # Uncomment this line if you want to display the plot interactively as well

# --- Main script logic ---
if __name__ == "__main__":
    # Define the problem sizes (number of nodes)
    problem_sizes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    num_runs_per_size = 3      # Number of times to execute each QUBO for averaging

    # Dictionary to store all results: {num_nodes: [time1, time2, ...]}
    results = {}
    # Lists to store data specifically for plotting
    plot_nodes_list = []
    plot_avg_times_list = []


    print("Starting Quanfluence performance measurement...")
    for num_nodes in problem_sizes:
        print(f"\n--- Testing with {num_nodes} nodes ---")
        
        try:
            # 1. Create the QUBO dictionary
            qubo_dict = create_max_cut_qubo(num_nodes)
            
            # Save QUBO to a temporary file for uploading
            temp_filename = f"max_cut_{num_nodes}_nodes.qubo"
            with open(temp_filename, 'w') as f:
                bqm = dimod.BQM.from_qubo(qubo_dict) # Use BQM to get num_variables
                f.write(f"p qubo {bqm.num_variables} {len(qubo_dict)}\n")
                for (i, j), value in qubo_dict.items():
                    f.write(f"{i} {j} {value}\n")
            print(f"Generated local QUBO file: {temp_filename}")

            # 2. Upload the QUBO file
            print(f"Uploading '{temp_filename}' to the server...")
            upload_response = client.upload_device_qubo(device_id, temp_filename)
            
            # Check if upload was successful and "result" key exists
            if "result" not in upload_response:
                print(f"Error uploading QUBO for {num_nodes} nodes. Response: {upload_response}")
                os.remove(temp_filename) # Clean up local file
                continue # Skip to next problem size
            
            uploaded_filename = upload_response["result"]
            print(f"QUBO uploaded successfully. Server filename: {uploaded_filename}")

            # Clean up the local temporary file
            os.remove(temp_filename)

            # 3. Execute the uploaded QUBO multiple times
            times_for_current_size = []
            for i in range(num_runs_per_size):
                print(f"  Running iteration {i+1}/{num_runs_per_size} for {num_nodes} nodes...")
                start_time = time.time()
                execution_result = client.execute_device_qubo_file(device_id, uploaded_filename)
                end_time = time.time()
                elapsed_time = end_time - start_time
                times_for_current_size.append(elapsed_time)
                print(f"  Iteration {i+1} completed in {elapsed_time:.4f} seconds.")
                # print(f"  Result (first 100 chars): {str(execution_result)[:100]}...") # For debugging

            results[num_nodes] = times_for_current_size
            avg_time = sum(times_for_current_size) / len(times_for_current_size)
            print(f"Average time for {num_nodes} nodes over {num_runs_per_size} runs: {avg_time:.4f} seconds")

            # Store data for plotting
            plot_nodes_list.append(num_nodes)
            plot_avg_times_list.append(avg_time)

        except FileNotFoundError:
            print(f"Error: The QUBO file '{temp_filename}' was not found for upload. This might happen if creation failed.")
        except ConnectionError as ce:
            print(f"A connection error occurred: {ce}. Please check your network and Quanfluence server status.")
            # Optionally, decide if you want to break or continue
            break 
        except Exception as e:
            print(f"An unexpected error occurred during processing for {num_nodes} nodes: {e}")
            # Optionally, clean up temp file if it exists and an error occurs mid-process
            if 'temp_filename' in locals() and os.path.exists(temp_filename):
                os.remove(temp_filename)
            continue # Continue to the next problem size if possible

    print("\n--- Summary of all collected execution times ---")
    if not results:
        print("No results collected. Cannot generate plot.")
    else:
        for num_nodes_summary, run_times_summary in results.items():
            avg_time_summary = sum(run_times_summary) / len(run_times_summary)
            print(f"Nodes: {num_nodes_summary}, Average Time: {avg_time_summary:.4f}s, Individual Times: {run_times_summary}")

        # --- Generate and save the plot ---
        # Ensure there's data to plot
        if plot_nodes_list and plot_avg_times_list:
            # Convert lists to numpy arrays for plotting function (as in original plot script)
            nodes_np_array = np.array(plot_nodes_list)
            avg_times_np_array = np.array(plot_avg_times_list)
            generate_and_save_plot(nodes_np_array, avg_times_np_array, filename="quanfluence_performance.png")
        else:
            print("No data available to generate a plot.")
            
    print("\nScript finished.")
