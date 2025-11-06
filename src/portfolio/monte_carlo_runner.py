"""
Monte Carlo Simulation Runner
Interactive script for running portfolio simulations
"""

import os
import glob
from .portfolio import Portfolio

def get_available_data_files():
    """Get list of available CSV data files"""
    # Get the script directory and find data folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check multiple possible data directory locations
    possible_dirs = [
        'data',  # Current directory
        '../data',  # Parent directory
        '../../data',  # Two levels up
        os.path.join(script_dir, '..', '..', '..', 'data'),  # Absolute path to MIAX/data
        'C:/Users/Iván/Desktop/Ivan/MIAX/data'  # Direct path
    ]
    
    data_dir = None
    for dir_path in possible_dirs:
        abs_path = os.path.abspath(dir_path)
        if os.path.exists(abs_path):
            data_dir = abs_path
            print(f"Found data directory: {data_dir}")
            break
    
    if not data_dir:
        print(f"Checked directories: {[os.path.abspath(d) for d in possible_dirs]}")
        return []
    
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    # Filter for price files only (exclude stats, income, balance, cashflow)
    price_files = [f for f in csv_files if all(x not in os.path.basename(f).lower() for x in ['stats', 'income', 'balance', 'cashflow'])]
    
    print(f"Total CSV files: {len(csv_files)}")
    print(f"Price files: {len(price_files)}")
    if csv_files:
        print("All files:")
        for f in csv_files:
            print(f"  {os.path.basename(f)}")
    if price_files:
        print("Price files:")
        for f in price_files:
            print(f"  {os.path.basename(f)}")
    
    return price_files

def parse_filename(filepath):
    """Extract asset symbol from filename"""
    filename = os.path.basename(filepath)
    # Remove extension and date suffix
    parts = filename.replace('.csv', '').split('_')
    return parts[0]  # First part is usually the symbol

def get_simulation_parameters():
    """Get Monte Carlo simulation parameters from user"""
    print("\n" + "=" * 50)
    print("MONTE CARLO SIMULATION PARAMETERS")
    print("=" * 50)
    
    # Days to simulate
    while True:
        try:
            days = int(input("Enter number of days to simulate (default 252): ") or "252")
            if days > 0:
                break
            else:
                print("Please enter a positive number")
        except ValueError:
            print("Please enter a valid number")
    
    # Number of simulations
    while True:
        try:
            simulations = int(input("Enter number of simulation paths (default 1000): ") or "1000")
            if simulations > 0:
                break
            else:
                print("Please enter a positive number")
        except ValueError:
            print("Please enter a valid number")
    
    # Initial portfolio value
    while True:
        try:
            initial_value = float(input("Enter initial portfolio value (default 10000): ") or "10000")
            if initial_value > 0:
                break
            else:
                print("Please enter a positive number")
        except ValueError:
            print("Please enter a valid number")
    
    # Simulation type
    print("\nSimulation type:")
    print("1. Portfolio as a whole")
    print("2. Individual assets")
    
    while True:
        try:
            sim_type = int(input("Enter choice (1-2): "))
            if sim_type in [1, 2]:
                break
            else:
                print("Please enter 1 or 2")
        except ValueError:
            print("Please enter a valid number")
    
    simulate_individual = (sim_type == 2)
    
    return days, simulations, initial_value, simulate_individual

def select_assets_and_weights():
    """Let user select assets and set weights"""
    available_files = get_available_data_files()
    
    if not available_files:
        print("No data files found in 'data' directory.")
        print("Please run the data downloader first.")
        return None, None
    
    print(f"\nAvailable assets ({len(available_files)} files found):")
    assets_map = {}
    
    for i, filepath in enumerate(available_files):
        symbol = parse_filename(filepath)
        assets_map[i] = (symbol, filepath)
        print(f"{i+1}. {symbol} ({os.path.basename(filepath)})")
    
    # Select assets
    print("\nSelect assets for portfolio:")
    selected_indices = input("Enter asset numbers (comma-separated, e.g., 1,2,3): ").strip()
    
    try:
        indices = [int(x.strip()) - 1 for x in selected_indices.split(',')]
        selected_assets = {assets_map[i][0]: assets_map[i][1] for i in indices if i in assets_map}
    except (ValueError, KeyError):
        print("Invalid selection")
        return None, None
    
    if not selected_assets:
        print("No valid assets selected")
        return None, None
    
    # Set weights
    print(f"\nSet weights for {len(selected_assets)} assets (must sum to 1.0):")
    weights = {}
    
    for asset in selected_assets.keys():
        while True:
            try:
                weight = float(input(f"Weight for {asset} (0.0-1.0): "))
                if 0 <= weight <= 1:
                    weights[asset] = weight
                    break
                else:
                    print("Weight must be between 0 and 1")
            except ValueError:
                print("Please enter a valid number")
    
    # Check if weights sum to 1
    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 0.01:
        print(f"Warning: Weights sum to {total_weight:.3f}, normalizing to 1.0")
        weights = {asset: w/total_weight for asset, w in weights.items()}
    
    return selected_assets, weights

def run_monte_carlo_simulation():
    """Main function to run Monte Carlo simulation"""
    
    print("=" * 60)
    print("PORTFOLIO MONTE CARLO SIMULATION")
    print("=" * 60)
    
    # Select assets and weights
    asset_files, weights = select_assets_and_weights()
    if not asset_files:
        return
    
    # Get simulation parameters
    days, simulations, initial_value, simulate_individual = get_simulation_parameters()
    
    try:
        # Create portfolio
        print(f"\n📊 Creating portfolio with {len(asset_files)} assets...")
        portfolio = Portfolio.from_csv_files(asset_files, weights)
        
        print(f"Portfolio composition:")
        for asset, weight in portfolio.weights.items():
            print(f"  {asset}: {weight:.1%}")
        
        # Run simulation
        print(f"\n🎲 Running Monte Carlo simulation...")
        print(f"  Days: {days}")
        print(f"  Simulations: {simulations}")
        print(f"  Initial value: ${initial_value:,.0f}")
        print(f"  Type: {'Individual assets' if simulate_individual else 'Portfolio'}")
        
        results = portfolio.monte_carlo_simulation(
            days=days,
            simulations=simulations,
            initial_value=initial_value,
            simulate_individual=simulate_individual
        )
        
        # Display results
        print(f"\n✅ Simulation complete!")
        
        if results['type'] == 'portfolio':
            final_values = results['final_values']
            print(f"\nPortfolio Results:")
            print(f"  Mean final value: ${final_values.mean():,.0f}")
            print(f"  Standard deviation: ${final_values.std():,.0f}")
            print(f"  5th percentile: ${final_values.min():,.0f}")
            print(f"  95th percentile: ${final_values.max():,.0f}")
            print(f"  Probability of loss: {(final_values < initial_value).mean():.1%}")
        else:
            print(f"\nIndividual Asset Results:")
            for asset, data in results['assets'].items():
                final_values = data['final_values']
                print(f"  {asset} (weight {data['weight']:.1%}):")
                print(f"    Mean final value: ${final_values.mean():,.0f}")
                print(f"    Probability of loss: {(final_values < initial_value * data['weight']).mean():.1%}")
        
        # Visualize results
        print(f"\n📈 Generating visualization...")
        portfolio.visualize_simulation(results, save_plot=True)
        
        print(f"\n🎯 Simulation complete! Check 'data/plots' folder for charts.")
        
    except Exception as e:
        print(f"\n❌ Error running simulation: {e}")

if __name__ == "__main__":
    run_monte_carlo_simulation()