import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import random

# Set random seed for reproducible results
RANDOM_SEED = 42  # Change this number for different but consistent results
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Global parameters
GRID_SIZE = 200  # Total grid area (200m x 200m)
BURDEN = 10.0    # meters - distance between rows (FIXED)
SPACING = 10.0   # meters - distance between columns (FIXED)

# Calculate rows and columns automatically from burden and spacing
ROWS = int(GRID_SIZE / BURDEN)      # 200 / 10 = 20 rows
COLUMNS = int(GRID_SIZE / SPACING)  # 200 / 10 = 20 columns
TOTAL_HOLES = ROWS * COLUMNS        # 20 * 20 = 400 holes

# Hardness levels
HARDNESS_LEVELS = {
    1: {'name': 'Soft', 'range': (1.0, 1.5), 'color': 'lightblue'},
    2: {'name': 'Medium', 'range': (1.6, 2.5), 'color': 'lightgreen'},
    3: {'name': 'Hard', 'range': (2.6, 3.5), 'color': 'orange'},
    4: {'name': 'Extra Hard', 'range': (3.6, 4.0), 'color': 'red'}
}

def create_hardness_zones():
    """Create 4 hardness areas based on predefined percentages"""
    # Calculate area sizes based on percentages
    soft_area = 0.05 * GRID_SIZE      # 5% of 200m = 10m
    medium_area = 0.15 * GRID_SIZE    # 15% of 200m = 30m  
    hard_area = 0.50 * GRID_SIZE      # 50% of 200m = 100m
    extra_hard_area = 0.30 * GRID_SIZE # 30% of 200m = 60m
    
    zones = [
        {
            'center_x': 100, 'center_y': GRID_SIZE - soft_area/2,  # 195m
            'width': 200, 'height': soft_area,
            'hardness': 1, 'name': 'Soft'
        },
        {
            'center_x': 100, 'center_y': GRID_SIZE - soft_area - medium_area/2,  # 175m
            'width': 200, 'height': medium_area,
            'hardness': 2, 'name': 'Medium'
        },
        {
            'center_x': 100, 'center_y': GRID_SIZE - soft_area - medium_area - hard_area/2,  # 110m
            'width': 200, 'height': hard_area,
            'hardness': 3, 'name': 'Hard'
        },
        {
            'center_x': 100, 'center_y': extra_hard_area/2,  # 30m
            'width': 200, 'height': extra_hard_area,
            'hardness': 4, 'name': 'Extra Hard'
        }
    ]
    return zones

def get_hardness_for_position(x, y, zones):
    """Get hardness level for a position based on predefined area percentages"""
    # Define hardness areas based on percentages
    # Soft: 0-5%, Medium: 10-20%, Hard: 40-60%, Extra Hard: 20-40%
    # Total: 5% + 15% + 50% + 30% = 100%
    
    # Calculate area boundaries based on percentages
    soft_area = 0.05 * GRID_SIZE      # 5% of 200m = 10m
    medium_area = 0.15 * GRID_SIZE    # 15% of 200m = 30m  
    hard_area = 0.50 * GRID_SIZE      # 50% of 200m = 100m
    extra_hard_area = 0.30 * GRID_SIZE # 30% of 200m = 60m
    
    # Define area boundaries (from top to bottom)
    area_boundaries = [
        {
            'y_min': GRID_SIZE - soft_area, 'y_max': GRID_SIZE,  # 190-200m
            'hardness': 1, 'name': 'Soft Area'
        },
        {
            'y_min': GRID_SIZE - soft_area - medium_area, 'y_max': GRID_SIZE - soft_area,  # 160-190m
            'hardness': 2, 'name': 'Medium Area'
        },
        {
            'y_min': GRID_SIZE - soft_area - medium_area - hard_area, 'y_max': GRID_SIZE - soft_area - medium_area,  # 60-160m
            'hardness': 3, 'name': 'Hard Area'
        },
        {
            'y_min': 0, 'y_max': GRID_SIZE - soft_area - medium_area - hard_area,  # 0-60m
            'hardness': 4, 'name': 'Extra Hard Area'
        }
    ]
    
    # Determine which area the point belongs to
    for area in area_boundaries:
        if area['y_min'] < y <= area['y_max']:
            # Return the area's specific hardness level
            return area['hardness']
    
    # Default to top area if y > GRID_SIZE
    return area_boundaries[0]['hardness']

def create_regular_grid():
    """Create regular grid with random hole generation (not all positions)"""
    holes = []
    zones = create_hardness_zones()
    
    # Define hole generation probability (0.0 to 1.0)
    HOLE_PROBABILITY = 0.6  # 60% chance of generating a hole at each position
    
    for row in range(ROWS):
        for col in range(COLUMNS):
            # Random decision: generate hole or not
            if random.random() < HOLE_PROBABILITY:
                # Calculate exact grid position
                x = col * SPACING + SPACING / 2  # Center of each cell
                y = row * BURDEN + BURDEN / 2    # Center of each cell
                
                # Assign hardness based on nearest zone
                hardness = get_hardness_for_position(x, y, zones)
                
                hole = {
                    'x': x,
                    'y': y,
                    'row': row + 1,  # Start from 1
                    'col': col + 1,  # Start from 1
                    'hardness': hardness
                }
                holes.append(hole)
    
    return holes, zones

def visualize_regular_grid(holes, zones):
    """Visualize regular grid with layered hardness zones and row/column labels"""
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Draw layered hardness zones (rectangular layers) for 20x20 grid
    for zone in zones:
        rect = patches.Rectangle((0, zone['center_y'] - zone['height']/2), 
                               200, zone['height'],
                               facecolor=HARDNESS_LEVELS[zone['hardness']]['color'],
                               alpha=0.3, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        
    
    # Draw straight boundaries between areas
    boundary_y_values = [190, 160, 60]  # Soft-Medium, Medium-Hard, Hard-Extra Hard boundaries
    
    for y_boundary in boundary_y_values:
        ax.axhline(y=y_boundary, color='black', linewidth=3, alpha=1.0)
    
    
    # Group holes by hardness
    hardness_groups = {1: [], 2: [], 3: [], 4: []}
    for hole in holes:
        hardness_groups[hole['hardness']].append(hole)
    
    # Plot holes for each hardness level
    for hardness, hole_list in hardness_groups.items():
        if hole_list:
            x_coords = [hole['x'] for hole in hole_list]
            y_coords = [hole['y'] for hole in hole_list]
            color = HARDNESS_LEVELS[hardness]['color']
            name = HARDNESS_LEVELS[hardness]['name']
            
            ax.scatter(x_coords, y_coords, c=color, s=120, 
                      edgecolors='black', linewidth=2, alpha=0.9,
                      label=f'{name} ({len(hole_list)} holes)')
            
            # Add row,column labels
            for hole in hole_list:
                ax.annotate(f"{hole['row']},{hole['col']}", 
                           (hole['x'], hole['y']), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=6, color='black', fontweight='bold')
    
    # Draw grid lines over holes
    for i in range(ROWS + 1):
        y_line = i * BURDEN
        ax.axhline(y=y_line, color='black', linestyle='-', alpha=0.7, linewidth=1)
        if i < ROWS:
            ax.text(-5, y_line + BURDEN/2, f'R{i+1}', 
                   ha='right', va='center', fontsize=10, color='blue', fontweight='bold')
    
    for j in range(COLUMNS + 1):
        x_line = j * SPACING
        ax.axvline(x=x_line, color='black', linestyle='-', alpha=0.7, linewidth=1)
        if j < COLUMNS:
            ax.text(x_line + SPACING/2, -5, f'C{j+1}', 
                   ha='center', va='top', fontsize=10, color='red', fontweight='bold')
    
    ax.set_xlim(-10, COLUMNS * SPACING + 10)
    ax.set_ylim(-10, ROWS * BURDEN + 10)
    ax.set_xlabel('X Coordinate (m)', fontsize=12)
    ax.set_ylabel('Y Coordinate (m)', fontsize=12)
    ax.set_title(f'Drill and Blast Point Generation with Hardness Levels\nBurden: {BURDEN}m, Spacing: {SPACING}m, Grid: {ROWS}×{COLUMNS}', 
                fontsize=16, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

def visualize_heatmap(holes, zones):
    """Visualize hardness data as heatmap"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create grid for interpolation
    x_coords = [hole['x'] for hole in holes]
    y_coords = [hole['y'] for hole in holes]
    hardness_values = [hole['hardness'] for hole in holes]
    
    # Create interpolation grid
    xi = np.linspace(0, 200, 100)
    yi = np.linspace(0, 200, 100)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # Interpolate hardness values
    from scipy.interpolate import griddata
    Zi = griddata((x_coords, y_coords), hardness_values, (Xi, Yi), method='nearest')
    
    # Create heatmap with exact 4 levels
    levels = [0.5, 1.5, 2.5, 3.5, 4.5]  # 4 hardness levels: 1, 2, 3, 4
    im = ax.contourf(Xi, Yi, Zi, levels=levels, colors=['lightblue', 'lightgreen', 'orange', 'red'], alpha=0.7)
    
    # Add drill holes
    for hole in holes:
        color = HARDNESS_LEVELS[hole['hardness']]['color']
        ax.scatter(hole['x'], hole['y'], c=color, s=50, edgecolors='black', linewidth=1, alpha=0.9)
    
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 200)
    ax.set_xlabel('X Coordinate (m)', fontsize=12)
    ax.set_ylabel('Y Coordinate (m)', fontsize=12)
    ax.set_title('Hardness Distribution Heatmap\nBurden: 10m, Spacing: 10m', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Hardness Level', fontsize=12)
    cbar.set_ticks([1, 2, 3, 4])
    cbar.set_ticklabels(['Soft', 'Medium', 'Hard', 'Extra Hard'])
    cbar.set_ticks([1, 2, 3, 4])  # Ensure only 4 ticks
    
    plt.tight_layout()
    plt.show()

def visualize_3d_surface(holes, zones):
    """Visualize hardness data as 3D surface"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create grid for interpolation
    x_coords = [hole['x'] for hole in holes]
    y_coords = [hole['y'] for hole in holes]
    hardness_values = [hole['hardness'] for hole in holes]
    
    # Create interpolation grid
    xi = np.linspace(0, 200, 50)
    yi = np.linspace(0, 200, 50)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # Interpolate hardness values
    from scipy.interpolate import griddata
    Zi = griddata((x_coords, y_coords), hardness_values, (Xi, Yi), method='nearest')
    
    # Create 3D surface
    surf = ax.plot_surface(Xi, Yi, Zi, cmap='viridis', alpha=0.8, linewidth=0, antialiased=True)
    
    # Add drill holes as 3D points
    for hole in holes:
        color = HARDNESS_LEVELS[hole['hardness']]['color']
        ax.scatter(hole['x'], hole['y'], hole['hardness'], c=color, s=50, alpha=0.9)
    
    ax.set_xlabel('X Coordinate (m)', fontsize=12)
    ax.set_ylabel('Y Coordinate (m)', fontsize=12)
    ax.set_zlabel('Hardness Level', fontsize=12)
    ax.set_title('3D Hardness Surface\nBurden: 10m, Spacing: 10m', fontsize=14, fontweight='bold')
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20)
    
    plt.tight_layout()
    plt.show()

def export_regular_grid_to_excel(holes, filename):
    """Export regular grid data to Excel"""
    # Hole data
    hole_data = []
    for i, hole in enumerate(holes):
        hardness_name = HARDNESS_LEVELS[hole['hardness']]['name']
        hardness_range = HARDNESS_LEVELS[hole['hardness']]['range']
        hardness_value = random.uniform(hardness_range[0], hardness_range[1])
        
        hole_data.append({
            'Hole_ID': f'H_{i+1:03d}',
            'X_Coordinate': round(hole['x'], 2),
            'Y_Coordinate': round(hole['y'], 2),
            'Row': hole['row'],
            'Column': hole['col'],
            'Hardness_Level': hole['hardness'],
            'Hardness_Name': hardness_name,
            'Hardness_Value': round(hardness_value, 2)
        })
    
    # Write to Excel (only Hole Data sheet)
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        pd.DataFrame(hole_data).to_excel(writer, sheet_name='Hole_Data', index=False)

def main():
    """Main function"""
    print("🚀 Zone-Based Grid Hardness Generator")
    print("="*50)
    
    
    # Create regular grid with zones
    holes, zones = create_regular_grid()
    
    # Convert to tuple format for compatibility
    holes_tuples = [(hole['x'], hole['y'], hole['hardness']) for hole in holes]
    
    # Create clusters by hardness level
    clusters = {1: [], 2: [], 3: [], 4: []}
    for hole in holes_tuples:
        clusters[hole[2]].append(hole)
    
    # Convert to list format
    clusters_list = [clusters[i] for i in range(1, 5)]
    
    # Print essential results
    print("\n📊 Generation Results:")
    for level in range(1, 5):
        level_holes = [hole for hole in holes_tuples if hole[2] == level]
        hardness_name = HARDNESS_LEVELS[level]['name']
        print(f"   {hardness_name:12}: {len(level_holes):2d} holes")
    
    print(f"\n✅ Total: {len(holes_tuples)} holes generated from {ROWS}x{COLUMNS} possible positions")
    print(f"📏 Grid Parameters:")
    print(f"   Grid Size: {GRID_SIZE}m x {GRID_SIZE}m")
    print(f"   Burden: {BURDEN}m (FIXED)")
    print(f"   Spacing: {SPACING}m (FIXED)")
    print(f"   Rows: {ROWS} (calculated from {GRID_SIZE}/{BURDEN})")
    print(f"   Columns: {COLUMNS} (calculated from {GRID_SIZE}/{SPACING})")
    print(f"   Possible Positions: {ROWS * COLUMNS}")
    print(f"   Hole Generation: Random (60% probability)")
    print(f"   Hardness Assignment: Percentage-based areas")
    print(f"🎯 Hardness Areas: Soft(5%), Medium(15%), Hard(50%), Extra Hard(30%)")
    
    
    # Visualize
    print("\n🎨 Generating visualizations...")
    visualize_regular_grid(holes, zones)
    visualize_heatmap(holes, zones)
    visualize_3d_surface(holes, zones)
    
    # Export to Excel
    export_regular_grid_to_excel(holes, "zone_based_grid_hardness_data.xlsx")
    print("📊 Data exported to: zone_based_grid_hardness_data.xlsx")

if __name__ == "__main__":
    main()
