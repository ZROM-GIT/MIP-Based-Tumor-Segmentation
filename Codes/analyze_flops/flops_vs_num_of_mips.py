import matplotlib.pyplot as plt

# Input data
num_mips = [16, 32, 48, 64, 80]
tflops = [0.3235, 0.647, 0.9705, 1.294, 1.6175]
inference_time = [0.233, 0.458 , 0.696, 0.913, 1.146]

# Create the plot
fig, ax1 = plt.subplots()

color1 = 'tab:blue'
ax1.set_xlabel('Number of MIPs')
ax1.set_ylabel('TFLOPs', color=color1)
ax1.plot(num_mips, tflops, marker='o', color=color1, label='TFLOPs')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(0.220, 1.650)

# Set x-axis ticks exactly at your specified MIP counts
ax1.set_xticks(num_mips)

# Create a second y-axis
ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel('Inference Time (seconds)', color=color2)
ax2.plot(num_mips, inference_time, marker='s', linestyle='--', color=color2, label='Inference Time')
ax2.tick_params(axis='y', labelcolor=color2)

# Title and grid
plt.title('TFLOPs and Inference Time vs. Number of MIPs')
fig.tight_layout()
plt.grid(True)

# Optional: add legend
fig.legend(loc='upper left', bbox_to_anchor=(0.118, 0.92), fontsize=11)

# Save figure
plt.savefig('flops_vs_num_of_mips.png', dpi=1200, bbox_inches='tight')

# Show plot
plt.show()
