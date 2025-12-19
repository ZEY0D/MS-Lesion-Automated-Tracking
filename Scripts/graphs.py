import matplotlib.pyplot as plt
import numpy as np

# --- DATA FROM YOUR VALIDATED "SENSIBLE" RUN ---
# (Based on the 7 New, 12 Gone, 4 Enlarged, 4 Shrunk findings)
categories = ['New', 'Disappeared', 'Enlarged', 'Shrunk', 'Stable']
counts = [7, 12, 4, 4, 5] # Assuming ~5 stable based on total count
colors = ['#ff4444', '#44ff44', '#ffaa00', '#00aaff', '#888888']
# Red=Bad, Green=Good, Orange=Growing, Blue=Healing, Grey=Stable

def generate_charts():
    print("📊 Generating Clinical Summary Charts...")

    # --- FIGURE 1: DISEASE ACTIVITY BAR CHART ---
    plt.figure(figsize=(10, 6))
    bars = plt.bar(categories, counts, color=colors, edgecolor='black', alpha=0.8)
    
    # Add numbers on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                 f'{int(height)}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.title('Patient 11: Disease Progression Summary', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Lesions', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Save
    plt.savefig('Final_Clinical_BarChart.png', dpi=300)
    print("✅ Saved: Final_Clinical_BarChart.png")
    plt.show()

    # --- FIGURE 2: LESION FATE PIE CHART ---
    # Combine "Good" outcomes vs "Bad" outcomes for a simpler view
    labels_pie = ['Active Disease\n(New + Enlarged)', 'Healing/Stable\n(Gone + Shrunk + Stable)']
    sizes_pie = [7+4, 12+4+5] # Bad vs Good
    colors_pie = ['#ff6666', '#66ff66']
    explode = (0.05, 0) # Pop out the "Active" slice slightly

    plt.figure(figsize=(8, 8))
    plt.pie(sizes_pie, explode=explode, labels=labels_pie, colors=colors_pie,
            autopct='%1.1f%%', shadow=True, startangle=140, textprops={'fontsize': 12, 'weight': 'bold'})
    
    plt.title('Overall Treatment Response', fontsize=14, fontweight='bold')
    
    # Save
    plt.savefig('Final_Clinical_PieChart.png', dpi=300)
    print("✅ Saved: Final_Clinical_PieChart.png")
    plt.show()

if __name__ == "__main__":
    generate_charts()