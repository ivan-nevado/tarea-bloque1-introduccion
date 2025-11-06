import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import os
from typing import Dict, List, Optional, Tuple

class VisualizationEngine:
    """Advanced visualization engine for portfolio analysis"""
    
    def __init__(self, portfolio):
        self.portfolio = portfolio
        self.plots_dir = 'data/plots'
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_price_evolution(self) -> str:
        """Plot normalized price evolution of all assets"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Normalize all prices to start at 100
        for asset, data in self.portfolio.assets.items():
            if 'close' in data.columns and len(data) > 0:
                normalized_prices = (data['close'] / data['close'].iloc[0]) * 100
                ax.plot(data.index, normalized_prices, label=asset, linewidth=2)
        
        ax.set_title('Portfolio Assets - Normalized Price Evolution', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Normalized Price (Base = 100)', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        
        filename = f'{self.plots_dir}/price_evolution.png'
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        return filename
    
    def plot_returns_distribution(self) -> str:
        """Plot returns distribution with risk metrics"""
        returns = self.portfolio.get_returns()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Individual asset returns distribution
        ax1 = axes[0, 0]
        for asset in returns.columns:
            asset_returns = returns[asset].dropna()
            if len(asset_returns) > 0:
                ax1.hist(asset_returns * 100, bins=50, alpha=0.6, label=asset, density=True)
        ax1.set_title('Daily Returns Distribution', fontweight='bold')
        ax1.set_xlabel('Daily Return (%)')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Portfolio returns distribution
        ax2 = axes[0, 1]
        portfolio_returns = self.portfolio.get_portfolio_returns()
        if len(portfolio_returns) > 0:
            ax2.hist(portfolio_returns * 100, bins=50, alpha=0.7, color='darkblue', density=True)
            ax2.axvline(portfolio_returns.mean() * 100, color='red', linestyle='--', label=f'Mean: {portfolio_returns.mean()*100:.2f}%')
            ax2.axvline(portfolio_returns.quantile(0.05) * 100, color='orange', linestyle='--', label=f'VaR 95%: {portfolio_returns.quantile(0.05)*100:.2f}%')
        ax2.set_title('Portfolio Returns Distribution', fontweight='bold')
        ax2.set_xlabel('Daily Return (%)')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Rolling volatility
        ax3 = axes[1, 0]
        if len(portfolio_returns) > 30:
            rolling_vol = portfolio_returns.rolling(30).std() * np.sqrt(252) * 100
            ax3.plot(rolling_vol.index, rolling_vol, color='purple', linewidth=2)
            ax3.fill_between(rolling_vol.index, rolling_vol, alpha=0.3, color='purple')
        ax3.set_title('30-Day Rolling Volatility', fontweight='bold')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Annualized Volatility (%)')
        ax3.grid(True, alpha=0.3)
        
        # Drawdown analysis
        ax4 = axes[1, 1]
        if len(portfolio_returns) > 0:
            cumulative = (1 + portfolio_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max * 100
            ax4.fill_between(drawdown.index, drawdown, 0, alpha=0.6, color='red')
            ax4.plot(drawdown.index, drawdown, color='darkred', linewidth=1)
        ax4.set_title('Portfolio Drawdown', fontweight='bold')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Drawdown (%)')
        ax4.grid(True, alpha=0.3)
        
        filename = f'{self.plots_dir}/returns_analysis.png'
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        return filename
    
    def plot_correlation_heatmap(self) -> str:
        """Plot correlation matrix heatmap"""
        returns = self.portfolio.get_returns()
        
        if len(returns.columns) < 2:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        correlation_matrix = returns.corr()
        
        # Create heatmap
        sns.heatmap(correlation_matrix, 
                   annot=True, 
                   cmap='RdYlBu_r', 
                   center=0,
                   square=True,
                   fmt='.3f',
                   cbar_kws={'label': 'Correlation Coefficient'},
                   ax=ax)
        
        ax.set_title('Asset Correlation Matrix', fontsize=16, fontweight='bold')
        
        filename = f'{self.plots_dir}/correlation_heatmap.png'
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        return filename
    
    def plot_risk_return_scatter(self) -> str:
        """Plot risk-return scatter with efficient frontier concept"""
        returns = self.portfolio.get_returns()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Calculate risk-return for each asset
        risk_return_data = []
        for asset in returns.columns:
            asset_returns = returns[asset].dropna()
            if len(asset_returns) > 0:
                annual_return = asset_returns.mean() * 252 * 100
                annual_vol = asset_returns.std() * np.sqrt(252) * 100
                weight = self.portfolio.weights.get(asset, 0)
                
                # Size bubble by weight
                size = weight * 1000 + 100
                ax.scatter(annual_vol, annual_return, s=size, alpha=0.7, label=asset)
                
                # Add asset labels
                ax.annotate(asset, (annual_vol, annual_return), 
                           xytext=(5, 5), textcoords='offset points', fontsize=10)
                
                risk_return_data.append((asset, annual_vol, annual_return, weight))
        
        # Add portfolio point
        portfolio_returns = self.portfolio.get_portfolio_returns()
        if len(portfolio_returns) > 0:
            port_return = portfolio_returns.mean() * 252 * 100
            port_vol = portfolio_returns.std() * np.sqrt(252) * 100
            ax.scatter(port_vol, port_return, s=500, marker='*', 
                      color='red', label='Portfolio', edgecolors='black', linewidth=2)
        
        ax.set_xlabel('Annualized Volatility (%)', fontsize=12)
        ax.set_ylabel('Annualized Return (%)', fontsize=12)
        ax.set_title('Risk-Return Analysis', fontsize=16, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Add risk-free line (assuming 2% risk-free rate)
        if len(risk_return_data) > 0:
            max_vol = max([x[1] for x in risk_return_data])
            x_line = np.linspace(0, max_vol, 100)
            y_line = 2 + (x_line * 0.3)  # Simple risk premium line
            ax.plot(x_line, y_line, '--', color='gray', alpha=0.5, label='Risk Premium Guide')
        
        filename = f'{self.plots_dir}/risk_return_scatter.png'
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        return filename
    
    def plot_portfolio_composition(self) -> str:
        """Plot portfolio composition pie chart and weight evolution"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Pie chart of current weights
        weights = list(self.portfolio.weights.values())
        labels = list(self.portfolio.weights.keys())
        colors = plt.cm.Set3(np.linspace(0, 1, len(weights)))
        
        wedges, texts, autotexts = ax1.pie(weights, labels=labels, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
        ax1.set_title('Portfolio Composition', fontsize=14, fontweight='bold')
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        # Weight comparison bar chart
        ax2.barh(labels, weights, color=colors)
        ax2.set_xlabel('Weight')
        ax2.set_title('Asset Weights', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add weight values on bars
        for i, (label, weight) in enumerate(zip(labels, weights)):
            ax2.text(weight + 0.01, i, f'{weight:.1%}', va='center', fontweight='bold')
        
        filename = f'{self.plots_dir}/portfolio_composition.png'
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        return filename
    
    def plot_monte_carlo_results(self, simulation_results: Dict) -> str:
        """Plot Monte Carlo simulation results with confidence intervals"""
        if simulation_results['type'] != 'portfolio':
            return None
        
        paths = simulation_results['paths']
        days = simulation_results['days']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Monte Carlo paths
        sample_size = min(200, simulation_results['simulations'])
        sample_indices = np.random.choice(simulation_results['simulations'], sample_size, replace=False)
        
        for i in sample_indices:
            ax1.plot(range(days + 1), paths[i], alpha=0.05, color='blue')
        
        # Percentile bands
        percentiles = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
        
        ax1.plot(range(days + 1), percentiles[2], 'r-', linewidth=3, label='Median (50th percentile)')
        ax1.fill_between(range(days + 1), percentiles[0], percentiles[4], alpha=0.2, color='gray', label='5th-95th percentile')
        ax1.fill_between(range(days + 1), percentiles[1], percentiles[3], alpha=0.3, color='blue', label='25th-75th percentile')
        
        ax1.set_title(f'Monte Carlo Simulation - {simulation_results["simulations"]} Paths', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Days')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Final value distribution
        final_values = simulation_results['final_values']
        ax2.hist(final_values, bins=50, alpha=0.7, color='darkgreen', density=True)
        
        # Add statistical lines
        ax2.axvline(final_values.mean(), color='red', linestyle='-', linewidth=2, label=f'Mean: ${final_values.mean():,.0f}')
        ax2.axvline(np.percentile(final_values, 5), color='orange', linestyle='--', linewidth=2, label=f'5th percentile: ${np.percentile(final_values, 5):,.0f}')
        ax2.axvline(np.percentile(final_values, 95), color='green', linestyle='--', linewidth=2, label=f'95th percentile: ${np.percentile(final_values, 95):,.0f}')
        
        ax2.set_title('Final Portfolio Value Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Final Portfolio Value ($)')
        ax2.set_ylabel('Probability Density')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        filename = f'{self.plots_dir}/monte_carlo_results.png'
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        return filename
    
    def plot_performance_metrics(self) -> str:
        """Plot key performance metrics dashboard"""
        returns = self.portfolio.get_returns()
        portfolio_returns = self.portfolio.get_portfolio_returns()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 1. Cumulative returns
        ax1 = axes[0, 0]
        for asset in returns.columns:
            asset_returns = returns[asset].dropna()
            if len(asset_returns) > 0:
                cumulative = (1 + asset_returns).cumprod()
                ax1.plot(cumulative.index, cumulative, label=asset, alpha=0.7)
        
        if len(portfolio_returns) > 0:
            port_cumulative = (1 + portfolio_returns).cumprod()
            ax1.plot(port_cumulative.index, port_cumulative, 'k-', linewidth=3, label='Portfolio')
        
        ax1.set_title('Cumulative Returns', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Rolling Sharpe ratio
        ax2 = axes[0, 1]
        if len(portfolio_returns) > 60:
            rolling_sharpe = (portfolio_returns.rolling(60).mean() * 252) / (portfolio_returns.rolling(60).std() * np.sqrt(252))
            ax2.plot(rolling_sharpe.index, rolling_sharpe, color='purple', linewidth=2)
            ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Sharpe = 1')
        ax2.set_title('60-Day Rolling Sharpe Ratio', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Monthly returns heatmap
        ax3 = axes[0, 2]
        if len(portfolio_returns) > 0:
            monthly_returns = portfolio_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            monthly_returns_pivot = monthly_returns.groupby([monthly_returns.index.year, monthly_returns.index.month]).first().unstack()
            
            if not monthly_returns_pivot.empty:
                sns.heatmap(monthly_returns_pivot * 100, annot=True, fmt='.1f', cmap='RdYlGn', center=0, ax=ax3)
        ax3.set_title('Monthly Returns Heatmap (%)', fontweight='bold')
        
        # 4. Asset contribution to portfolio return
        ax4 = axes[1, 0]
        contributions = []
        labels = []
        for asset, weight in self.portfolio.weights.items():
            if asset in returns.columns:
                asset_returns = returns[asset].dropna()
                if len(asset_returns) > 0:
                    contribution = asset_returns.mean() * 252 * weight * 100
                    contributions.append(contribution)
                    labels.append(asset)
        
        if contributions:
            colors = ['green' if x > 0 else 'red' for x in contributions]
            ax4.barh(labels, contributions, color=colors, alpha=0.7)
        ax4.set_title('Asset Contribution to Portfolio Return (%)', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')
        
        # 5. Risk contribution
        ax5 = axes[1, 1]
        risk_contributions = []
        for asset, weight in self.portfolio.weights.items():
            if asset in returns.columns:
                asset_returns = returns[asset].dropna()
                if len(asset_returns) > 0:
                    risk_contrib = (asset_returns.std() * np.sqrt(252) * weight) * 100
                    risk_contributions.append(risk_contrib)
        
        if risk_contributions:
            ax5.pie(risk_contributions, labels=labels, autopct='%1.1f%%', startangle=90)
        ax5.set_title('Risk Contribution', fontweight='bold')
        
        # 6. Performance summary table
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # Create performance summary
        summary_data = []
        if len(portfolio_returns) > 0:
            annual_return = portfolio_returns.mean() * 252
            annual_vol = portfolio_returns.std() * np.sqrt(252)
            sharpe = annual_return / annual_vol if annual_vol > 0 else 0
            
            # Max drawdown
            cumulative = (1 + portfolio_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_dd = drawdown.min()
            
            summary_data = [
                ['Annual Return', f'{annual_return:.2%}'],
                ['Annual Volatility', f'{annual_vol:.2%}'],
                ['Sharpe Ratio', f'{sharpe:.3f}'],
                ['Max Drawdown', f'{max_dd:.2%}'],
                ['Total Days', f'{len(portfolio_returns)}'],
                ['Positive Days', f'{(portfolio_returns > 0).sum()} ({(portfolio_returns > 0).mean():.1%})']
            ]
        
        if summary_data:
            table = ax6.table(cellText=summary_data, 
                            colLabels=['Metric', 'Value'],
                            cellLoc='center',
                            loc='center',
                            bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
        
        ax6.set_title('Performance Summary', fontweight='bold')
        
        filename = f'{self.plots_dir}/performance_dashboard.png'
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        return filename