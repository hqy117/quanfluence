#!/usr/bin/env python3
"""
Gset Problem Comparison: Quanfluence vs Simulated Annealing (dwave-neal)

This script compares the execution speed and solution quality between:
1. Quanfluence quantum annealing platform
2. dwave-neal simulated annealing

The script uses Gset benchmark problems for Max-Cut optimization.
"""

import time
import os
import sys
import dimod
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import neal  # dwave-neal for simulated annealing
from quanfluence_sdk import QuanfluenceClient

class GsetComparison:
    def __init__(self):
        """Initialize the comparison framework."""
        self.setup_quanfluence()
        self.setup_simulated_annealing()
        self.results = {
            'quanfluence': {},
            'simulated_annealing': {}
        }
        
    def setup_quanfluence(self):
        """Setup Quanfluence client."""
        try:
            self.qf_client = QuanfluenceClient()
            self.qf_client.signin('rochester_user0', 'Rochesteruser@123')
            self.device_id = 16
            print("✓ Quanfluence client initialized successfully")
        except Exception as e:
            print(f"✗ Error initializing Quanfluence client: {e}")
            sys.exit(1)
            
    def setup_simulated_annealing(self):
        """Setup dwave-neal simulated annealing sampler."""
        try:
            self.sa_sampler = neal.SimulatedAnnealingSampler()
            print("✓ Simulated Annealing sampler initialized successfully")
        except Exception as e:
            print(f"✗ Error initializing SA sampler: {e}")
            sys.exit(1)
    
    def parse_gset_file(self, filename):
        """
        Parse a Gset file and return the graph as edge list.
        
        Args:
            filename (str): Path to the Gset file
            
        Returns:
            tuple: (num_nodes, edges) where edges is list of (u, v, weight)
        """
        edges = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            
        # First line contains number of nodes and edges
        first_line = lines[0].strip().split()
        num_nodes = int(first_line[0])
        num_edges = int(first_line[1])
        
        # Parse edges (1-indexed in file, convert to 0-indexed)
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) >= 3:
                u = int(parts[0]) - 1  # Convert to 0-indexed
                v = int(parts[1]) - 1  # Convert to 0-indexed
                weight = float(parts[2])
                edges.append((u, v, weight))
                
        print(f"Parsed Gset file: {num_nodes} nodes, {len(edges)} edges")
        return num_nodes, edges
    
    def create_max_cut_qubo(self, num_nodes, edges):
        """
        Create QUBO formulation for Max-Cut problem.
        
        Args:
            num_nodes (int): Number of nodes in the graph
            edges (list): List of (u, v, weight) tuples
            
        Returns:
            dict: QUBO dictionary
        """
        Q = {}
        
        # Max-Cut QUBO formulation:
        # Maximize: sum_{(i,j) in E} w_ij * (x_i + x_j - 2*x_i*x_j)
        # Convert to minimization: minimize -sum_{(i,j) in E} w_ij * (x_i + x_j - 2*x_i*x_j)
        
        for u, v, weight in edges:
            # Quadratic term: +2*w_ij for x_i*x_j (since we're minimizing the negative)
            Q[(u, v)] = Q.get((u, v), 0.0) + 2.0 * weight
            
            # Linear terms: -w_ij for x_i and x_j
            Q[(u, u)] = Q.get((u, u), 0.0) - weight
            Q[(v, v)] = Q.get((v, v), 0.0) - weight
        
        # Ensure all values are float32 for Quanfluence compatibility
        for key in Q:
            Q[key] = np.float32(Q[key])
            
        return Q
    
    def calculate_cut_value(self, solution, edges):
        """
        Calculate the cut value for a given solution.
        
        Args:
            solution (dict): Binary solution {node: 0/1}
            edges (list): List of (u, v, weight) tuples
            
        Returns:
            float: Cut value
        """
        cut_value = 0.0
        for u, v, weight in edges:
            if u in solution and v in solution:
                if solution[u] != solution[v]:  # Edge is cut
                    cut_value += weight
        return cut_value
    
    def run_quanfluence(self, qubo_dict, problem_name, num_runs=3):
        """
        Run the QUBO problem on Quanfluence platform.
        
        Args:
            qubo_dict (dict): QUBO formulation
            problem_name (str): Name for the problem
            num_runs (int): Number of runs for averaging
            
        Returns:
            dict: Results including timing and solution quality
        """
        print(f"\n--- Running {problem_name} on Quanfluence ---")
        
        # Create temporary QUBO file
        temp_filename = f"{problem_name}_quanfluence.qubo"
        bqm = dimod.BQM.from_qubo(qubo_dict)
        
        with open(temp_filename, 'w') as f:
            f.write(f"p qubo {bqm.num_variables} {len(qubo_dict)}\n")
            for (i, j), value in qubo_dict.items():
                f.write(f"{i} {j} {value}\n")
        
        try:
            # Upload QUBO using the already signed-in client
            print(f"Uploading {temp_filename}...")
            upload_response = self.qf_client.upload_device_qubo(self.device_id, temp_filename)
            
            if "result" not in upload_response:
                raise Exception(f"Upload failed: {upload_response}")
                
            uploaded_filename = upload_response["result"]
            print(f"✓ Uploaded as: {uploaded_filename}")
            
            # Run multiple times for timing
            execution_times = []
            solutions = []
            
            for run in range(num_runs):
                print(f"  Run {run + 1}/{num_runs}...")
                start_time = time.time()
                result = self.qf_client.execute_device_qubo_file(self.device_id, uploaded_filename)
                end_time = time.time()
                
                execution_time = end_time - start_time
                execution_times.append(execution_time)
                solutions.append(result)
                
                # Extract and display energy if available
                energy_str = ""
                if isinstance(result, dict) and 'energy' in result:
                    energy_str = f", Energy: {result['energy']:.2f}"
                
                print(f"  ✓ Completed in {execution_time:.4f}s{energy_str}")
            
            # Clean up
            os.remove(temp_filename)
            
            return {
                'execution_times': execution_times,
                'avg_time': np.mean(execution_times),
                'std_time': np.std(execution_times),
                'solutions': solutions,
                'success': True
            }
            
        except Exception as e:
            print(f"✗ Quanfluence execution failed: {e}")
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            return {
                'execution_times': [],
                'avg_time': float('inf'),
                'std_time': 0,
                'solutions': [],
                'success': False,
                'error': str(e)
            }
    
    def run_simulated_annealing(self, qubo_dict, problem_name, num_runs=3, num_reads=1000):
        """
        Run the QUBO problem using dwave-neal Simulated Annealing.
        
        Args:
            qubo_dict (dict): QUBO formulation
            problem_name (str): Name for the problem
            num_runs (int): Number of runs for averaging
            num_reads (int): Number of SA reads per run
            
        Returns:
            dict: Results including timing and solution quality
        """
        print(f"\n--- Running {problem_name} on Simulated Annealing ---")
        
        try:
            bqm = dimod.BQM.from_qubo(qubo_dict)
            execution_times = []
            solutions = []
            cut_values = []
            
            for run in range(num_runs):
                print(f"  Run {run + 1}/{num_runs}...")
                start_time = time.time()
                
                # Run simulated annealing
                sampleset = self.sa_sampler.sample(bqm, num_reads=num_reads)
                
                end_time = time.time()
                execution_time = end_time - start_time
                execution_times.append(execution_time)
                
                # Get best solution
                best_sample = sampleset.first.sample
                best_energy = sampleset.first.energy
                
                solutions.append({
                    'sample': best_sample,
                    'energy': best_energy,
                    'sampleset': sampleset
                })
                
                print(f"  ✓ Completed in {execution_time:.4f}s, Best energy: {best_energy:.2f}")
            
            return {
                'execution_times': execution_times,
                'avg_time': np.mean(execution_times),
                'std_time': np.std(execution_times),
                'solutions': solutions,
                'success': True
            }
            
        except Exception as e:
            print(f"✗ Simulated Annealing execution failed: {e}")
            return {
                'execution_times': [],
                'avg_time': float('inf'),
                'std_time': 0,
                'solutions': [],
                'success': False,
                'error': str(e)
            }
    
    def compare_gset_problem(self, gset_file, num_runs=3):
        """
        Compare Quanfluence and SA on a specific Gset problem.
        
        Args:
            gset_file (str): Path to Gset file
            num_runs (int): Number of runs for each method
            
        Returns:
            dict: Comparison results
        """
        problem_name = os.path.basename(gset_file).replace('.txt', '')
        print(f"\n{'='*60}")
        print(f"COMPARING PROBLEM: {problem_name}")
        print(f"{'='*60}")
        
        # Parse the Gset file
        num_nodes, edges = self.parse_gset_file(gset_file)
        
        # Create QUBO formulation
        qubo_dict = self.create_max_cut_qubo(num_nodes, edges)
        print(f"QUBO size: {len(qubo_dict)} terms")
        
        # Run on both platforms
        qf_results = self.run_quanfluence(qubo_dict, problem_name, num_runs)
        sa_results = self.run_simulated_annealing(qubo_dict, problem_name, num_runs)
        
        # Store results
        comparison = {
            'problem_name': problem_name,
            'num_nodes': num_nodes,
            'num_edges': len(edges),
            'quanfluence': qf_results,
            'simulated_annealing': sa_results,
            'edges': edges  # Store for cut value calculation
        }
        
        # Calculate and compare cut values if both succeeded
        if qf_results['success'] and sa_results['success']:
            self.analyze_solution_quality(comparison)
        
        return comparison
    
    def analyze_solution_quality(self, comparison):
        """Analyze and compare solution quality between methods."""
        edges = comparison['edges']
        
        # Analyze SA solutions (we have the actual solutions)
        sa_cut_values = []
        sa_energies = []
        for sol in comparison['simulated_annealing']['solutions']:
            cut_value = self.calculate_cut_value(sol['sample'], edges)
            sa_cut_values.append(cut_value)
            sa_energies.append(sol['energy'])
        
        comparison['simulated_annealing']['cut_values'] = sa_cut_values
        comparison['simulated_annealing']['best_cut'] = max(sa_cut_values)
        comparison['simulated_annealing']['avg_cut'] = np.mean(sa_cut_values)
        comparison['simulated_annealing']['energies'] = sa_energies
        comparison['simulated_annealing']['best_energy'] = min(sa_energies)
        comparison['simulated_annealing']['avg_energy'] = np.mean(sa_energies)
        
        # Analyze Quanfluence solutions
        qf_cut_values = []
        qf_energies = []
        for sol in comparison['quanfluence']['solutions']:
            if isinstance(sol, dict) and 'result' in sol and 'energy' in sol:
                # Convert Quanfluence format: {'0': -1.0, '1': 1.0, ...} to {0: 0, 1: 1, ...}
                # Quanfluence uses -1/+1, we need 0/1 for cut calculation
                qf_solution = {}
                for var_str, value in sol['result'].items():
                    var_int = int(var_str)
                    # Convert -1/+1 to 0/1: -1 -> 0, +1 -> 1
                    qf_solution[var_int] = 0 if value < 0 else 1
                
                cut_value = self.calculate_cut_value(qf_solution, edges)
                qf_cut_values.append(cut_value)
                qf_energies.append(sol['energy'])
        
        if qf_cut_values:
            comparison['quanfluence']['cut_values'] = qf_cut_values
            comparison['quanfluence']['best_cut'] = max(qf_cut_values)
            comparison['quanfluence']['avg_cut'] = np.mean(qf_cut_values)
            comparison['quanfluence']['energies'] = qf_energies
            comparison['quanfluence']['best_energy'] = min(qf_energies)
            comparison['quanfluence']['avg_energy'] = np.mean(qf_energies)
        
        print(f"\nSolution Quality Analysis:")
        print(f"SA - Best cut: {max(sa_cut_values):.2f}, Avg cut: {np.mean(sa_cut_values):.2f}")
        print(f"SA - Best energy: {min(sa_energies):.2f}, Avg energy: {np.mean(sa_energies):.2f}")
        
        if qf_cut_values:
            print(f"QF - Best cut: {max(qf_cut_values):.2f}, Avg cut: {np.mean(qf_cut_values):.2f}")
            print(f"QF - Best energy: {min(qf_energies):.2f}, Avg energy: {np.mean(qf_energies):.2f}")
            
            # Compare best solutions
            if max(qf_cut_values) > max(sa_cut_values):
                print("→ Quanfluence found better cut value!")
            elif max(qf_cut_values) < max(sa_cut_values):
                print("→ Simulated Annealing found better cut value!")
            else:
                print("→ Both methods found same best cut value!")
        else:
            print("QF - Could not parse solution format")
    
    def run_comparison_suite(self, gset_files, num_runs=3):
        """
        Run comparison on multiple Gset problems.
        
        Args:
            gset_files (list): List of Gset file paths
            num_runs (int): Number of runs per problem per method
        """
        all_results = []
        
        for gset_file in gset_files:
            try:
                result = self.compare_gset_problem(gset_file, num_runs)
                all_results.append(result)
                self.print_comparison_summary(result)
            except Exception as e:
                print(f"✗ Failed to process {gset_file}: {e}")
                continue
        
        # Generate overall analysis
        self.generate_performance_analysis(all_results)
        return all_results
    
    def print_comparison_summary(self, result):
        """Print a summary of the comparison for one problem."""
        print(f"\n--- SUMMARY: {result['problem_name']} ---")
        print(f"Problem size: {result['num_nodes']} nodes, {result['num_edges']} edges")
        
        qf = result['quanfluence']
        sa = result['simulated_annealing']
        
        if qf['success']:
            print(f"Quanfluence: {qf['avg_time']:.4f}s ± {qf['std_time']:.4f}s")
            if 'best_cut' in qf:
                print(f"  QF Best Cut: {qf['best_cut']:.2f}, Best Energy: {qf['best_energy']:.2f}")
        else:
            print(f"Quanfluence: FAILED - {qf.get('error', 'Unknown error')}")
            
        if sa['success']:
            print(f"Simulated Annealing: {sa['avg_time']:.4f}s ± {sa['std_time']:.4f}s")
            if 'best_cut' in sa:
                print(f"  SA Best Cut: {sa['best_cut']:.2f}, Best Energy: {sa['best_energy']:.2f}")
        else:
            print(f"Simulated Annealing: FAILED - {sa.get('error', 'Unknown error')}")
        
        if qf['success'] and sa['success']:
            speedup = sa['avg_time'] / qf['avg_time']
            print(f"Speedup (SA time / QF time): {speedup:.2f}x")
            
            # Quality comparison
            if 'best_cut' in qf and 'best_cut' in sa:
                if qf['best_cut'] > sa['best_cut']:
                    print("→ Quanfluence found better solution quality!")
                elif qf['best_cut'] < sa['best_cut']:
                    print("→ SA found better solution quality!")
                else:
                    print("→ Both found same solution quality!")
    
    def generate_performance_analysis(self, all_results):
        """Generate comprehensive performance analysis and plots."""
        print(f"\n{'='*60}")
        print("OVERALL PERFORMANCE ANALYSIS")
        print(f"{'='*60}")
        
        # Extract data for plotting
        problem_names = []
        node_counts = []
        qf_times = []
        sa_times = []
        successful_problems = []
        
        for result in all_results:
            if result['quanfluence']['success'] and result['simulated_annealing']['success']:
                successful_problems.append(result)
                problem_names.append(result['problem_name'])
                node_counts.append(result['num_nodes'])
                qf_times.append(result['quanfluence']['avg_time'])
                sa_times.append(result['simulated_annealing']['avg_time'])
        
        if not successful_problems:
            print("No successful comparisons to analyze.")
            return
        
        # Create performance comparison plot
        self.create_performance_plots(problem_names, node_counts, qf_times, sa_times)
        
        # Print statistics
        print(f"\nSuccessful comparisons: {len(successful_problems)}/{len(all_results)}")
        print(f"Average Quanfluence time: {np.mean(qf_times):.4f}s")
        print(f"Average SA time: {np.mean(sa_times):.4f}s")
        print(f"Average speedup (SA/QF): {np.mean(np.array(sa_times)/np.array(qf_times)):.2f}x")
    
    def create_performance_plots(self, problem_names, node_counts, qf_times, sa_times):
        """Create performance comparison plots."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Execution times by problem
        x_pos = np.arange(len(problem_names))
        width = 0.35
        
        ax1.bar(x_pos - width/2, qf_times, width, label='Quanfluence', alpha=0.8)
        ax1.bar(x_pos + width/2, sa_times, width, label='Simulated Annealing', alpha=0.8)
        ax1.set_xlabel('Problem')
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.set_title('Execution Time Comparison by Problem')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(problem_names, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Execution times vs problem size
        ax2.scatter(node_counts, qf_times, label='Quanfluence', alpha=0.7, s=50)
        ax2.scatter(node_counts, sa_times, label='Simulated Annealing', alpha=0.7, s=50)
        ax2.set_xlabel('Number of Nodes')
        ax2.set_ylabel('Execution Time (seconds)')
        ax2.set_title('Execution Time vs Problem Size')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Speedup analysis
        speedups = np.array(sa_times) / np.array(qf_times)
        ax3.bar(problem_names, speedups, alpha=0.8, color='green')
        ax3.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Equal Performance')
        ax3.set_xlabel('Problem')
        ax3.set_ylabel('Speedup (SA time / QF time)')
        ax3.set_title('Speedup Analysis (>1 means QF is faster)')
        ax3.set_xticklabels(problem_names, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Log-scale time comparison
        ax4.scatter(node_counts, qf_times, label='Quanfluence', alpha=0.7, s=50)
        ax4.scatter(node_counts, sa_times, label='Simulated Annealing', alpha=0.7, s=50)
        ax4.set_xlabel('Number of Nodes')
        ax4.set_ylabel('Execution Time (seconds, log scale)')
        ax4.set_title('Execution Time vs Problem Size (Log Scale)')
        ax4.set_yscale('log')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('gset_comparison_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Performance analysis plots saved to 'gset_comparison_analysis.png'")


def main():
    """Main execution function."""
    print("Gset Problem Comparison: Quanfluence vs Simulated Annealing")
    print("=" * 60)
    
    # Initialize comparison framework
    comparison = GsetComparison()
    
    # Select Gset problems to test (start with smaller ones)
    gset_files = [
        'Gset/G14.txt',  # 800 nodes - medium size
        'Gset/G15.txt',  # Similar size
        'Gset/G11.txt',  # Smaller problem
        'Gset/G12.txt',  # Smaller problem
        'Gset/G13.txt',  # Smaller problem
    ]
    
    # Filter to only existing files
    existing_files = [f for f in gset_files if os.path.exists(f)]
    
    if not existing_files:
        print("No Gset files found. Please check the Gset directory.")
        return
    
    print(f"Testing {len(existing_files)} Gset problems:")
    for f in existing_files:
        print(f"  - {f}")
    
    # Run the comparison
    results = comparison.run_comparison_suite(existing_files, num_runs=3)
    
    print(f"\n{'='*60}")
    print("COMPARISON COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved and analysis plots generated.")


if __name__ == "__main__":
    main() 