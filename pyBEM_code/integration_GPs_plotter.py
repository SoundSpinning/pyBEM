import os
import sys
import numpy as np

from utils import (
    get_tri_points,
    get_tri_points_7p,
    get_quad_points,
    get_quad_points_3x3,
    split_quad_get_quad_points_3x3
)

def plot_point_sequence(ax, points, color='#DC143C', linestyle=':', alpha=0.6):
    """Draws dotted arrows between sequential Gauss points."""
    pts = np.asarray(points)
    for i in range(len(pts) - 1):
        x1, y1 = pts[i, 0], pts[i, 1]
        x2, y2 = pts[i + 1, 0], pts[i + 1, 1]
        
        dx = x2 - x1
        dy = y2 - y1
        dist = np.hypot(dx, dy)
        if dist > 0:
            x1_s = x1 + (dx / dist) * 1.5
            y1_s = y1 + (dy / dist) * 1.5
            x2_s = x2 - (dx / dist) * 1.5
            y2_s = y2 - (dy / dist) * 1.5
            
            ax.annotate(
                "",
                xy=(x2_s, y2_s), xycoords='data',
                xytext=(x1_s, y1_s), textcoords='data',
                arrowprops=dict(
                    arrowstyle="->",
                    linestyle=linestyle,
                    color=color,
                    lw=1.2,
                    alpha=alpha,
                    mutation_scale=12
                )
            )

def draw_element_outline_cw(ax, vertices, labels):
    """
    Draws element edges with Clockwise directional arrows matching the reference diagram
    (Right-Hand Rule normal pointing AWAY from viewer, -Z).
    """
    pts = np.asarray(vertices)
    num_v = len(pts)
    
    for i in range(num_v):
        p1 = pts[i]
        p2 = pts[(i + 1) % num_v]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', lw=2)
        
        # Add directional arrows along element edges
        mx, my = (p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0
        dx, dy = (p2[0] - p1[0]), (p2[1] - p1[1])
        ax.annotate(
            "",
            xy=(mx + dx*0.05, my + dy*0.05), xycoords='data',
            xytext=(mx - dx*0.05, my - dy*0.05), textcoords='data',
            arrowprops=dict(arrowstyle="->", color='blue', lw=2, mutation_scale=15)
        )

    # Plot vertex points and text labels
    for pt, label in zip(pts, labels):
        ax.plot(pt[0], pt[1], 'bo', ms=8)
        ha = 'right' if pt[0] <= 25 else 'left'
        va = 'top' if pt[1] == 0 else 'bottom'
        y_offset = -3.5 if pt[1] == 0 else 2.0
        ax.text(pt[0], pt[1] + y_offset, f"{label}\n({pt[0]:.0f}, {pt[1]:.0f})", 
                color='blue', fontweight='bold', ha=ha, va=va, fontsize=8)

def generate_integration_gps_poster(output_dir="."):
    """
    Generates 'pyBEM_integration_GPs.png' with consistent Clockwise node ordering 
    and matching Gauss point sequence paths across all panels.
    Optimized for high-performance non-interactive rendering.
    """
    # 1. Force Agg backend safety check
    if 'matplotlib.pyplot' not in sys.modules:
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Clear lingering figure objects from memory
    plt.close('all')

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=120)
    fig.suptitle("pyBEM Acoustic Solver - Integration Schemes (Normal Away from Viewer -Z)", 
                 fontsize=14, fontweight='bold')

    # Nodal definitions
    v1_tri = np.array([0.0, 0.0, 0.0])
    v2_tri = np.array([0.0, 50.0, 0.0])
    v3_tri = np.array([50.0, 0.0, 0.0])

    v1_q = np.array([0.0, 0.0, 0.0])
    v2_q = np.array([0.0, 50.0, 0.0])
    v3_q = np.array([50.0, 50.0, 0.0])
    v4_q = np.array([50.0, 0.0, 0.0])

    tri_verts = [v1_tri, v2_tri, v3_tri]
    tri_labels = ["v1", "v3", "v2"]

    quad_verts = [v1_q, v2_q, v3_q, v4_q]
    quad_labels = ["v1", "v2", "v3", "v4"]

    # --- Helper: Fast Batch Annotator ---
    def annotate_points(ax, pts, start_idx=1, color='#DC143C', custom_offsets=None):
        pts = np.asarray(pts)
        # Vectorized point plot
        ax.scatter(pts[:, 0], pts[:, 1], c=color, s=40, zorder=4)
        
        # Batch text placement
        for i, (x, y) in enumerate(pts[:, :2]):
            idx = start_idx + i
            y_off = custom_offsets.get(idx, 1.8) if custom_offsets else 1.8
            ax.text(x, y + y_off, f"P{idx}\n({x:.1f}, {y:.1f})", 
                    fontsize=6.5, fontweight='bold', ha='center', va='bottom')

    # 1. TRIA_3gp (Mid-Order)
    ax = axes[0, 0]
    draw_element_outline_cw(ax, tri_verts, tri_labels)
    gps1, _ = get_tri_points(v1_tri, v2_tri, v3_tri)
    plot_point_sequence(ax, gps1, color='#DC143C')
    annotate_points(ax, gps1, start_idx=1, color='#DC143C')
    ax.set_title("Tri 3-Point (Mid-Order)", fontweight='bold', fontsize=10)

    # 2. TRIA_7gp (High-Order)
    ax = axes[0, 1]
    draw_element_outline_cw(ax, tri_verts, tri_labels)
    gps2, _ = get_tri_points_7p(v1_tri, v2_tri, v3_tri)
    plot_point_sequence(ax, gps2, color='#DC143C')
    annotate_points(ax, gps2, start_idx=1, color='#DC143C', custom_offsets={3: -4.5})
    ax.set_title("Tri 7-Point (High-Order)", fontweight='bold', fontsize=10)

    # 3. QUAD_4gp (Mid-Order)
    ax = axes[0, 2]
    draw_element_outline_cw(ax, quad_verts, quad_labels)
    gps3, _ = get_quad_points(v1_q, v2_q, v3_q, v4_q)
    plot_point_sequence(ax, gps3, color='#1E90FF')
    annotate_points(ax, gps3, start_idx=1, color='#1E90FF')
    ax.set_title("Quad 4-Point (2x2 Mid-Order)", fontweight='bold', fontsize=10)

    # 4. QUAD_9gp (3x3 High-Order)
    ax = axes[1, 0]
    draw_element_outline_cw(ax, quad_verts, quad_labels)
    gps4, _ = get_quad_points_3x3(v1_q, v2_q, v3_q, v4_q)
    plot_point_sequence(ax, gps4, color='#1E90FF')
    annotate_points(ax, gps4, start_idx=1, color='#1E90FF')
    ax.set_title("Quad 9-Point (3x3 High-Order)", fontweight='bold', fontsize=10)

    # 5. SPLIT QUAD (2 x TRIA_7gp)
    ax = axes[1, 1]
    draw_element_outline_cw(ax, quad_verts, quad_labels)
    ax.plot([0, 50], [0, 50], 'b--', lw=1.5, alpha=0.7)

    gps5, _ = split_quad_get_quad_points_3x3(v1_q, v2_q, v3_q, v4_q)
    gps5 = np.asarray(gps5)
    gps5_t1, gps5_t2 = gps5[:7], gps5[7:]

    plot_point_sequence(ax, gps5_t1, color='#DC143C')
    plot_point_sequence(ax, gps5_t2, color='#FF8C00')

    annotate_points(ax, gps5_t1, start_idx=1, color='#DC143C', custom_offsets={7: -4.5})
    annotate_points(ax, gps5_t2, start_idx=8, color='#FF8C00', custom_offsets={12: -4.5})

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#DC143C', ms=7, label='Tri 1 (v1,v2,v3)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF8C00', ms=7, label='Tri 2 (v1,v3,v4)')
    ]
    ax.legend(handles=legend_elements, loc='upper center', fontsize=7.5, framealpha=0.9)
    ax.set_title("pyBEM Sub-Triangulation (2x TRIA_7gp)", fontweight='bold', fontsize=10)

    # Apply shared axis formatting cleanly across active subplots
    for row in range(2):
        for col in range(3):
            if row == 1 and col == 2:
                continue
            a = axes[row, col]
            a.grid(True, linestyle=":", alpha=0.6)
            a.set_xlim(-10, 60)
            a.set_ylim(-10, 60)
            a.set_aspect('equal')

    # Hide unused panel (bottom right)
    axes[1, 2].axis('off')

    # Fast fixed margin adjustment (Replaces slow tight_layout)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.05, hspace=0.22, wspace=0.20)

    output_path = os.path.join(output_dir, "pyBEM_integration_GPs.png")
    plt.savefig(output_path, dpi=120)
    plt.close(fig)