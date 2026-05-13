import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("final_plots", exist_ok=True)

# =========================
# 1. Fusion comparison chart
# =========================
methods = ["Single-tag", "Naive fusion", "Filtered fusion"]
radial_spans = [20.70, 220.31, 11.92]

plt.figure(figsize=(7, 5))
plt.bar(methods, radial_spans)
plt.ylabel("Radial Span (mm)")
plt.title("Localization Stability Under Viewpoint Changes")
plt.grid(axis="y", alpha=0.3)
for i, v in enumerate(radial_spans):
    plt.text(i, v + 5, f"{v:.2f} mm", ha="center")
plt.tight_layout()
plt.savefig("final_plots/fusion_comparison_bar.png", dpi=300)
plt.close()


# ===================================
# 2. Per-tag viewpoint drift bar chart
# ===================================
tags = ["Tag 0", "Tag 1", "Tag 2", "Tag 3", "Tag 10", "Tag 13"]
tag_radial_spans = [16.15, 14.52, 24.23, 43.58, 62.07, 21.25]

plt.figure(figsize=(8, 5))
plt.bar(tags, tag_radial_spans)
plt.ylabel("Origin-frame Radial Span (mm)")
plt.title("Fixed-Tag Coordinate Drift Under Viewpoint Changes")
plt.grid(axis="y", alpha=0.3)
for i, v in enumerate(tag_radial_spans):
    plt.text(i, v + 1, f"{v:.1f}", ha="center")
plt.tight_layout()
plt.savefig("final_plots/per_tag_viewpoint_drift.png", dpi=300)
plt.close()


# ========================================
# 3. Single-tag vs filtered fusion drift
# ========================================
single = np.array([
    [169.4, 317.3, 9.4],
    [168.6, 318.2, 8.2],
    [169.2, 322.1, 5.9],
    [167.7, 326.3, 1.3],
    [175.2, 323.8, 18.3],
])

filtered = np.array([
    [173.1, 209.9, 21.3],
    [173.4, 208.8, 22.3],
    [171.6, 215.0, 18.3],
    [169.4, 217.1, 15.7],
    [172.4, 217.8, 17.1],
])

view_idx = np.arange(1, len(single) + 1)

single_drift = np.linalg.norm(single - single.mean(axis=0), axis=1)
filtered_drift = np.linalg.norm(filtered - filtered.mean(axis=0), axis=1)

plt.figure(figsize=(8, 5))
plt.plot(view_idx, single_drift, marker="o", label="Single-tag")
plt.plot(view_idx, filtered_drift, marker="o", label="Filtered fusion")
plt.xlabel("Viewpoint Image Index")
plt.ylabel("Drift from Mean Position (mm)")
plt.title("Coordinate Drift Across Camera Viewpoints")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("final_plots/viewpoint_drift_line_plot.png", dpi=300)
plt.close()


# ================================
# 4. 3D coordinate scatter plot
# ================================
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(single[:, 0], single[:, 1], single[:, 2], marker="o", label="Single-tag")
ax.scatter(filtered[:, 0], filtered[:, 1], filtered[:, 2], marker="^", label="Filtered fusion")

ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
ax.set_zlabel("Z (mm)")
ax.set_title("3D Coordinate Spread Across Viewpoints")
ax.legend()
plt.tight_layout()
plt.savefig("final_plots/3d_coordinate_scatter.png", dpi=300)
plt.close()


# ===================================
# 5. Summary comparison table as plot
# ===================================
fig, ax = plt.subplots(figsize=(8, 2.5))
ax.axis("off")

table_data = [
    ["Single-tag", "20.70 mm", "Baseline"],
    ["Naive fusion", "220.31 mm", "Worse due to unreliable tags"],
    ["Filtered fusion", "11.92 mm", "Best / confidence-aware"],
]

table = ax.table(
    cellText=table_data,
    colLabels=["Method", "Radial Span", "Interpretation"],
    loc="center",
    cellLoc="center",
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

plt.title("Fusion Method Comparison")
plt.tight_layout()
plt.savefig("final_plots/fusion_results_table.png", dpi=300)
plt.close()


print("Saved plots to final_plots/")
print("Generated:")
print("  final_plots/fusion_comparison_bar.png")
print("  final_plots/per_tag_viewpoint_drift.png")
print("  final_plots/viewpoint_drift_line_plot.png")
print("  final_plots/3d_coordinate_scatter.png")
print("  final_plots/fusion_results_table.png")