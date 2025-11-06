import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def draw_architecture():
    """Draw the financial system architecture using matplotlib"""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Define colors for each layer
    colors = {
        'ui': '#e3f2fd',
        'orchestration': '#f3e5f5', 
        'downloader': '#e8f5e8',
        'core': '#fff3e0',
        'processor': '#fce4ec',
        'external': '#f1f8e9'
    }
    
    # Define component positions and properties
    components = [
        # User Interface Layer
        {'name': 'Main Interface\n(main.py)', 'pos': (5, 7.5), 'color': colors['ui'], 'id': 'main'},
        
        # Orchestration Layer
        {'name': 'Parallel Downloader\n(parallel_downloader.py)', 'pos': (2.5, 6.5), 'color': colors['orchestration'], 'id': 'parallel'},
        {'name': 'Monte Carlo Runner\n(monte_carlo_runner.py)', 'pos': (7.5, 6.5), 'color': colors['orchestration'], 'id': 'monte_carlo'},
        
        # Data Downloaders Layer
        {'name': 'Stock Prices\n(stock_prices.py)', 'pos': (1, 5.5), 'color': colors['downloader'], 'id': 'stock'},
        {'name': 'Index Prices\n(index_prices.py)', 'pos': (2.5, 5.5), 'color': colors['downloader'], 'id': 'index'},
        {'name': 'SimFin Downloader\n(simfin_downloader.py)', 'pos': (4, 5.5), 'color': colors['downloader'], 'id': 'simfin'},
        {'name': 'Alpha Vantage Base\n(alpha_vantage_base.py)', 'pos': (1.75, 4.5), 'color': colors['downloader'], 'id': 'av_base'},
        {'name': 'Statistics Analyzer\n(statistics_analyzer.py)', 'pos': (4, 4.5), 'color': colors['downloader'], 'id': 'stats'},
        
        # Core Analysis Layer
        {'name': 'Portfolio DataClass\n(portfolio.py)', 'pos': (7.5, 5.5), 'color': colors['core'], 'id': 'portfolio'},
        
        # Data Processing Layer
        {'name': 'Data Preprocessor\n(data_preprocessor.py)', 'pos': (6, 4.5), 'color': colors['processor'], 'id': 'preprocessor'},
        {'name': 'Visualization Engine\n(visualization_engine.py)', 'pos': (8.5, 4.5), 'color': colors['processor'], 'id': 'viz'},
        {'name': 'Flexible Loader\n(flexible_loader.py)', 'pos': (7.5, 3.5), 'color': colors['processor'], 'id': 'loader'},
        
        # External Services
        {'name': 'Alpha Vantage API', 'pos': (1.75, 3.5), 'color': colors['external'], 'id': 'av_api'},
        {'name': 'SimFin API', 'pos': (4, 3.5), 'color': colors['external'], 'id': 'simfin_api'},
        {'name': 'Data Storage\n(CSV Files)', 'pos': (2.5, 2.5), 'color': colors['external'], 'id': 'storage'},
    ]
    
    # Draw components
    component_boxes = {}
    for comp in components:
        box = FancyBboxPatch(
            (comp['pos'][0] - 0.6, comp['pos'][1] - 0.3),
            1.2, 0.6,
            boxstyle="round,pad=0.05",
            facecolor=comp['color'],
            edgecolor='black',
            linewidth=1
        )
        ax.add_patch(box)
        
        # Add text
        ax.text(comp['pos'][0], comp['pos'][1], comp['name'], 
                ha='center', va='center', fontsize=8, fontweight='bold')
        
        component_boxes[comp['id']] = comp['pos']
    
    # Define connections
    connections = [
        ('main', 'parallel'),
        ('main', 'monte_carlo'),
        ('parallel', 'stock'),
        ('parallel', 'index'),
        ('parallel', 'simfin'),
        ('stock', 'av_base'),
        ('index', 'av_base'),
        ('av_base', 'av_api'),
        ('simfin', 'simfin_api'),
        ('simfin', 'stats'),
        ('stock', 'storage'),
        ('index', 'storage'),
        ('simfin', 'storage'),
        ('monte_carlo', 'portfolio'),
        ('portfolio', 'preprocessor'),
        ('portfolio', 'viz'),
        ('portfolio', 'loader'),
        ('preprocessor', 'storage'),
        ('loader', 'storage'),
    ]
    
    # Draw connections
    for start, end in connections:
        start_pos = component_boxes[start]
        end_pos = component_boxes[end]
        
        arrow = ConnectionPatch(
            start_pos, end_pos, "data", "data",
            arrowstyle="->", shrinkA=30, shrinkB=30,
            mutation_scale=20, fc="gray", alpha=0.6
        )
        ax.add_patch(arrow)
    
    # Add title
    ax.text(5, 7.8, '🏦 Financial Data Analysis System Architecture', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Add layer labels
    layer_labels = [
        {'text': 'User Interface', 'pos': (0.5, 7.5), 'color': colors['ui']},
        {'text': 'Orchestration', 'pos': (0.5, 6.5), 'color': colors['orchestration']},
        {'text': 'Data Downloaders', 'pos': (0.5, 5), 'color': colors['downloader']},
        {'text': 'Core Analysis', 'pos': (0.5, 5.5), 'color': colors['core']},
        {'text': 'Data Processing', 'pos': (0.5, 4), 'color': colors['processor']},
        {'text': 'External Services', 'pos': (0.5, 3), 'color': colors['external']},
    ]
    
    for label in layer_labels:
        ax.text(label['pos'][0], label['pos'][1], label['text'], 
                ha='left', va='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=label['color'], alpha=0.7))
    
    # Add statistics
    stats_text = """
📊 System Statistics:
• 15 Components
• 19 Relationships  
• 6 Architectural Layers
• 7 Chart Types
• 60% Speed Improvement (Parallel)
• Rate Limited APIs (12s Alpha Vantage, 1s SimFin)
    """
    
    ax.text(9.5, 2, stats_text, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('financial_system_architecture.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    draw_architecture()