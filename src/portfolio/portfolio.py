import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import os
from datetime import datetime
from .data_preprocessor import DataPreprocessor
from .visualization_engine import VisualizationEngine

@dataclass
class Portfolio:
    """Portfolio DataClass with Monte Carlo simulation capabilities"""
    
    assets: Dict[str, pd.DataFrame]
    weights: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        """Initialize portfolio after creation"""
        if self.weights is None:
            # Equal weights if not specified
            n_assets = len(self.assets)
            self.weights = {asset: 1/n_assets for asset in self.assets.keys()}
        
        # Validate weights sum to 1
        if abs(sum(self.weights.values()) - 1.0) > 0.01:
            raise ValueError("Portfolio weights must sum to 1.0")
    
    def get_returns(self) -> pd.DataFrame:
        """Calculate daily returns for all assets with data validation"""
        returns = pd.DataFrame()
        
        for asset, data in self.assets.items():
            if 'close' in data.columns and len(data) > 1:
                asset_returns = data['close'].pct_change().dropna()
                # Remove extreme outliers
                asset_returns = asset_returns[asset_returns.abs() < 1.0]  # Remove >100% daily changes
                returns[asset] = asset_returns
        
        return returns
    
    def get_portfolio_returns(self) -> pd.Series:
        """Calculate weighted portfolio returns"""
        returns = self.get_returns()
        
        # Align dates across all assets
        returns = returns.dropna()
        
        # Calculate weighted returns
        portfolio_returns = pd.Series(0, index=returns.index)
        for asset, weight in self.weights.items():
            if asset in returns.columns:
                portfolio_returns += returns[asset] * weight
        
        return portfolio_returns
    
    def monte_carlo_simulation(self, 
                             days: int = 252, 
                             simulations: int = 1000,
                             initial_value: float = 10000,
                             simulate_individual: bool = False) -> Dict:
        """
        Run Monte Carlo simulation
        
        Args:
            days: Number of days to simulate
            simulations: Number of simulation paths
            initial_value: Starting portfolio value
            simulate_individual: If True, simulate each asset individually
        
        Returns:
            Dictionary with simulation results
        """
        
        if simulate_individual:
            return self._simulate_individual_assets(days, simulations, initial_value)
        else:
            return self._simulate_portfolio(days, simulations, initial_value)
    
    def _simulate_portfolio(self, days: int, simulations: int, initial_value: float) -> Dict:
        """Simulate portfolio as a whole"""
        portfolio_returns = self.get_portfolio_returns()
        
        if len(portfolio_returns) < 30:
            raise ValueError("Need at least 30 days of data for simulation")
        
        # Calculate statistics
        mean_return = portfolio_returns.mean()
        std_return = portfolio_returns.std()
        
        # Generate random returns
        np.random.seed(42)  # For reproducibility
        random_returns = np.random.normal(mean_return, std_return, (simulations, days))
        
        # Calculate cumulative values
        price_paths = np.zeros((simulations, days + 1))
        price_paths[:, 0] = initial_value
        
        for i in range(days):
            price_paths[:, i + 1] = price_paths[:, i] * (1 + random_returns[:, i])
        
        return {
            'type': 'portfolio',
            'paths': price_paths,
            'final_values': price_paths[:, -1],
            'mean_return': mean_return,
            'std_return': std_return,
            'days': days,
            'simulations': simulations,
            'initial_value': initial_value
        }
    
    def _simulate_individual_assets(self, days: int, simulations: int, initial_value: float) -> Dict:
        """Simulate each asset individually"""
        returns = self.get_returns()
        results = {}
        
        for asset in self.assets.keys():
            if asset not in returns.columns:
                continue
            
            asset_returns = returns[asset].dropna()
            
            if len(asset_returns) < 30:
                continue
            
            # Calculate statistics
            mean_return = asset_returns.mean()
            std_return = asset_returns.std()
            
            # Generate random returns
            np.random.seed(42)  # For reproducibility
            random_returns = np.random.normal(mean_return, std_return, (simulations, days))
            
            # Calculate cumulative values
            price_paths = np.zeros((simulations, days + 1))
            asset_initial_value = initial_value * self.weights.get(asset, 0)
            price_paths[:, 0] = asset_initial_value
            
            for i in range(days):
                price_paths[:, i + 1] = price_paths[:, i] * (1 + random_returns[:, i])
            
            results[asset] = {
                'paths': price_paths,
                'final_values': price_paths[:, -1],
                'mean_return': mean_return,
                'std_return': std_return,
                'weight': self.weights.get(asset, 0)
            }
        
        return {
            'type': 'individual',
            'assets': results,
            'days': days,
            'simulations': simulations,
            'initial_value': initial_value
        }
    
    def visualize_simulation(self, simulation_results: Dict, save_plot: bool = True):
        """Visualize Monte Carlo simulation results"""
        
        if simulation_results['type'] == 'portfolio':
            self._plot_portfolio_simulation(simulation_results, save_plot)
        else:
            self._plot_individual_simulation(simulation_results, save_plot)
    
    def _plot_portfolio_simulation(self, results: Dict, save_plot: bool):
        """Plot portfolio simulation results"""
        paths = results['paths']
        days = results['days']
        
        plt.figure(figsize=(12, 8))
        
        # Plot all paths (sample for performance)
        sample_size = min(100, results['simulations'])
        sample_indices = np.random.choice(results['simulations'], sample_size, replace=False)
        
        for i in sample_indices:
            plt.plot(range(days + 1), paths[i], alpha=0.1, color='blue')
        
        # Plot percentiles
        percentiles = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
        
        plt.plot(range(days + 1), percentiles[2], 'r-', linewidth=2, label='Median (50th percentile)')
        plt.fill_between(range(days + 1), percentiles[0], percentiles[4], alpha=0.2, color='gray', label='5th-95th percentile')
        plt.fill_between(range(days + 1), percentiles[1], percentiles[3], alpha=0.3, color='blue', label='25th-75th percentile')
        
        plt.title(f'Portfolio Monte Carlo Simulation ({results["simulations"]} paths, {days} days)')
        plt.xlabel('Days')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        final_values = results['final_values']
        plt.text(0.02, 0.98, f'Final Value Statistics:\n'
                              f'Mean: ${final_values.mean():,.0f}\n'
                              f'Std: ${final_values.std():,.0f}\n'
                              f'5th percentile: ${np.percentile(final_values, 5):,.0f}\n'
                              f'95th percentile: ${np.percentile(final_values, 95):,.0f}',
                 transform=plt.gca().transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if save_plot:
            os.makedirs('data/plots', exist_ok=True)
            plt.savefig(f'data/plots/portfolio_monte_carlo.png', dpi=300, bbox_inches='tight')
            print("📊 Portfolio simulation plot saved: data/plots/portfolio_monte_carlo.png")
        
        plt.show()
    
    def _plot_individual_simulation(self, results: Dict, save_plot: bool):
        """Plot individual asset simulation results"""
        assets = results['assets']
        n_assets = len(assets)
        
        if n_assets == 0:
            print("No assets to plot")
            return
        
        # Create subplots
        cols = min(2, n_assets)
        rows = (n_assets + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if n_assets == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, (asset, data) in enumerate(assets.items()):
            ax = axes[idx] if n_assets > 1 else axes[0]
            
            paths = data['paths']
            days = results['days']
            
            # Plot sample paths
            sample_size = min(50, results['simulations'])
            sample_indices = np.random.choice(results['simulations'], sample_size, replace=False)
            
            for i in sample_indices:
                ax.plot(range(days + 1), paths[i], alpha=0.1, color='blue')
            
            # Plot percentiles
            percentiles = np.percentile(paths, [5, 50, 95], axis=0)
            ax.plot(range(days + 1), percentiles[1], 'r-', linewidth=2, label='Median')
            ax.fill_between(range(days + 1), percentiles[0], percentiles[2], alpha=0.2, color='gray', label='5th-95th percentile')
            
            ax.set_title(f'{asset} (Weight: {data["weight"]:.1%})')
            ax.set_xlabel('Days')
            ax.set_ylabel('Value ($)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_assets, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_plot:
            os.makedirs('data/plots', exist_ok=True)
            plt.savefig(f'data/plots/individual_assets_monte_carlo.png', dpi=300, bbox_inches='tight')
            print("📊 Individual assets simulation plot saved: data/plots/individual_assets_monte_carlo.png")
        
        plt.show()
    
    @classmethod
    def from_csv_files(cls, file_paths: Dict[str, str], weights: Optional[Dict[str, float]] = None):
        """Create portfolio from CSV files with automatic data cleaning"""
        assets = {}
        
        for asset, file_path in file_paths.items():
            if os.path.exists(file_path):
                try:
                    # Use preprocessor to handle any format
                    df = DataPreprocessor.process_any_format(file_path)
                    assets[asset] = df
                    print(f"✓ {asset}: {len(df)} data points loaded and cleaned")
                except Exception as e:
                    print(f"✗ {asset}: Failed to process - {e}")
            else:
                print(f"Warning: File not found for {asset}: {file_path}")
        
        return cls(assets=assets, weights=weights)
    
    @classmethod
    def from_any_data(cls, data_sources: Dict[str, Union[pd.DataFrame, str, dict]], 
                     weights: Optional[Dict[str, float]] = None,
                     price_columns: Optional[Dict[str, str]] = None,
                     date_columns: Optional[Dict[str, str]] = None):
        """Create portfolio from any data format"""
        assets = {}
        
        for asset, data in data_sources.items():
            try:
                price_col = price_columns.get(asset) if price_columns else None
                date_col = date_columns.get(asset) if date_columns else None
                
                df = DataPreprocessor.process_any_format(data, price_col, date_col)
                assets[asset] = df
                print(f"✓ {asset}: {len(df)} data points processed")
            except Exception as e:
                print(f"✗ {asset}: Failed to process - {e}")
        
        return cls(assets=assets, weights=weights)
    
    def report(self, 
              include_monte_carlo: bool = True,
              monte_carlo_days: int = 252,
              monte_carlo_simulations: int = 1000,
              initial_value: float = 10000,
              save_to_file: bool = True) -> str:
        """
        Generate comprehensive portfolio analysis report in Markdown format
        
        Args:
            include_monte_carlo: Whether to run Monte Carlo simulation
            monte_carlo_days: Days to simulate
            monte_carlo_simulations: Number of simulation paths
            initial_value: Initial portfolio value for simulation
            save_to_file: Whether to save report to file
        
        Returns:
            Markdown formatted report string
        """
        
        report_lines = []
        warnings = []
        
        # Header
        report_lines.extend([
            "# Portfolio Analysis Report",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ])
        
        # Portfolio Overview
        report_lines.extend([
            "## Portfolio Overview",
            f"**Number of Assets:** {len(self.assets)}",
            ""
        ])
        
        # Asset composition table
        report_lines.append("| Asset | Weight | Data Points | Date Range |")
        report_lines.append("|-------|--------|-------------|------------|")
        
        total_weight = 0
        for asset, weight in self.weights.items():
            if asset in self.assets:
                data = self.assets[asset]
                data_points = len(data)
                date_range = f"{data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}"
                
                # Check for warnings
                if data_points < 100:
                    warnings.append(f"⚠️ {asset} has only {data_points} data points (recommended: >100)")
                if weight > 0.5:
                    warnings.append(f"⚠️ {asset} has high concentration ({weight:.1%}) - consider diversification")
                
                report_lines.append(f"| {asset} | {weight:.1%} | {data_points} | {date_range} |")
                total_weight += weight
        
        report_lines.append("")
        
        # Weight validation
        if abs(total_weight - 1.0) > 0.01:
            warnings.append(f"⚠️ Portfolio weights sum to {total_weight:.1%} instead of 100%")
        
        # Returns Analysis
        try:
            returns = self.get_returns()
            portfolio_returns = self.get_portfolio_returns()
            
            if not returns.empty and not portfolio_returns.empty:
                report_lines.extend([
                    "## Returns Analysis",
                    ""
                ])
                
                # Portfolio statistics
                port_mean = portfolio_returns.mean() * 252
                port_vol = portfolio_returns.std() * np.sqrt(252)
                port_sharpe = port_mean / port_vol if port_vol > 0 else 0
                
                report_lines.extend([
                    "### Portfolio Performance",
                    f"- **Annualized Return:** {port_mean:.2%}",
                    f"- **Annualized Volatility:** {port_vol:.2%}",
                    f"- **Sharpe Ratio:** {port_sharpe:.3f}",
                    ""
                ])
                
                # Performance warnings
                if port_sharpe < 0.5:
                    warnings.append(f"⚠️ Portfolio has low Sharpe ratio ({port_sharpe:.3f})")
                    
        except Exception as e:
            warnings.append(f"⚠️ Could not calculate returns analysis: {e}")
        
        # Monte Carlo Simulation
        if include_monte_carlo:
            try:
                report_lines.extend([
                    "## Monte Carlo Simulation",
                    f"**Parameters:** {monte_carlo_simulations} simulations, {monte_carlo_days} days, ${initial_value:,.0f} initial value",
                    ""
                ])
                
                results = self.monte_carlo_simulation(
                    days=monte_carlo_days,
                    simulations=monte_carlo_simulations,
                    initial_value=initial_value,
                    simulate_individual=False
                )
                
                final_values = results['final_values']
                
                # Simulation statistics
                mean_final = final_values.mean()
                percentiles = np.percentile(final_values, [5, 50, 95])
                prob_loss = (final_values < initial_value).mean()
                
                report_lines.extend([
                    "### Simulation Results",
                    f"- **Expected Final Value:** ${mean_final:,.0f}",
                    f"- **5th Percentile (VaR 95%):** ${percentiles[0]:,.0f}",
                    f"- **Median:** ${percentiles[1]:,.0f}",
                    f"- **95th Percentile:** ${percentiles[2]:,.0f}",
                    f"- **Probability of Loss:** {prob_loss:.1%}",
                    ""
                ])
                
                # Risk warnings from simulation
                if prob_loss > 0.3:
                    warnings.append(f"⚠️ High probability of loss ({prob_loss:.1%}) in simulation")
                    
            except Exception as e:
                warnings.append(f"⚠️ Could not run Monte Carlo simulation: {e}")
        
        # Warnings Section
        if warnings:
            report_lines.extend([
                "## ⚠️ Warnings and Recommendations",
                ""
            ])
            for warning in warnings:
                report_lines.append(f"- {warning}")
            report_lines.append("")
        
        # Footer
        report_lines.extend([
            "---",
            "*This report is for informational purposes only and should not be considered as investment advice.*"
        ])
        
        # Join all lines
        report_text = "\n".join(report_lines)
        
        # Save to file if requested
        if save_to_file:
            os.makedirs('data/reports', exist_ok=True)
            filename = f"data/reports/portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"📄 Report saved to: {filename}")
        
        return report_text
    
    def plots_report(self, 
                    include_monte_carlo: bool = True,
                    monte_carlo_days: int = 252,
                    monte_carlo_simulations: int = 1000,
                    initial_value: float = 10000) -> Dict[str, str]:
        """
        Generate comprehensive visual analysis with multiple chart types
        
        Args:
            include_monte_carlo: Whether to run Monte Carlo simulation for plots
            monte_carlo_days: Days to simulate
            monte_carlo_simulations: Number of simulation paths  
            initial_value: Initial portfolio value for simulation
        
        Returns:
            Dictionary mapping plot names to file paths
        """
        
        print("📈 Generating comprehensive portfolio visualizations...")
        
        viz_engine = VisualizationEngine(self)
        plot_files = {}
        
        # 1. Price Evolution Chart
        try:
            print("  • Creating price evolution chart...")
            plot_files['price_evolution'] = viz_engine.plot_price_evolution()
        except Exception as e:
            print(f"    ⚠️ Price evolution plot failed: {e}")
        
        # 2. Returns Analysis Dashboard  
        try:
            print("  • Creating returns analysis dashboard...")
            plot_files['returns_analysis'] = viz_engine.plot_returns_distribution()
        except Exception as e:
            print(f"    ⚠️ Returns analysis plot failed: {e}")
        
        # 3. Correlation Heatmap
        try:
            print("  • Creating correlation heatmap...")
            corr_file = viz_engine.plot_correlation_heatmap()
            if corr_file:
                plot_files['correlation_heatmap'] = corr_file
        except Exception as e:
            print(f"    ⚠️ Correlation heatmap failed: {e}")
        
        # 4. Risk-Return Scatter
        try:
            print("  • Creating risk-return analysis...")
            plot_files['risk_return_scatter'] = viz_engine.plot_risk_return_scatter()
        except Exception as e:
            print(f"    ⚠️ Risk-return scatter failed: {e}")
        
        # 5. Portfolio Composition
        try:
            print("  • Creating portfolio composition charts...")
            plot_files['portfolio_composition'] = viz_engine.plot_portfolio_composition()
        except Exception as e:
            print(f"    ⚠️ Portfolio composition plot failed: {e}")
        
        # 6. Performance Dashboard
        try:
            print("  • Creating performance metrics dashboard...")
            plot_files['performance_dashboard'] = viz_engine.plot_performance_metrics()
        except Exception as e:
            print(f"    ⚠️ Performance dashboard failed: {e}")
        
        # 7. Monte Carlo Results (if requested)
        if include_monte_carlo:
            try:
                print("  • Running Monte Carlo simulation and creating charts...")
                mc_results = self.monte_carlo_simulation(
                    days=monte_carlo_days,
                    simulations=monte_carlo_simulations,
                    initial_value=initial_value,
                    simulate_individual=False
                )
                
                mc_file = viz_engine.plot_monte_carlo_results(mc_results)
                if mc_file:
                    plot_files['monte_carlo_results'] = mc_file
            except Exception as e:
                print(f"    ⚠️ Monte Carlo visualization failed: {e}")
        
        # Summary
        print(f"\n🎨 Generated {len(plot_files)} visualization(s):")
        for plot_name, file_path in plot_files.items():
            print(f"  ✓ {plot_name}: {file_path}")
        
        print(f"\n📁 All plots saved to: {viz_engine.plots_dir}/")
        
        return plot_files