#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize mapping results from a placing.txt file.

Supported formats:
- 2D:
		Total Rows: 19
		Total Columns: 21
		tile: 0 location: [18, 10] layer: 1
		tile: 1 location: [17, 11]

- 3D:
		Total Rows: 19
		Total Columns: 21
		Total Tiers: 4
		tile: 0 location: [18, 10, 0] layer: 1
		tile: 1 location: [17, 11, 0]

Usage:
	python visualize.py --input ../Parameters/placing.txt --output ../Parameters/placing_map.png
	# For 3D, per-tier plotting:
	python visualize.py -i ../Parameters/placing.txt --tier 0
	python visualize.py -i ../Parameters/placing.txt --all-tiers
"""

from __future__ import annotations

import argparse
import os
import re
from typing import List, Tuple, Dict, Optional

import matplotlib.pyplot as plt
from matplotlib import patches, cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patheffects as patheffects
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def parse_placing(file_path: str) -> Tuple[int, int, List[Tuple[int, int, int]], Dict[int, int]]:
	"""Parse placing.txt and return (rows, cols, placements, tile_layers).

	- placements: list of tuples (tile_id, row, col)
	- tile_layers: mapping tile_id -> layer_id (if not present in file, will be empty)
	"""
	if not os.path.isfile(file_path):
		raise FileNotFoundError(f"placing file not found: {file_path}")

	rows = cols = None
	placements: List[Tuple[int, int, int]] = []
	tile_layers: Dict[int, int] = {}

	# Patterns
	row_pat = re.compile(r"^\s*Total\s+Rows:\s*(\d+)\s*$", re.IGNORECASE)
	col_pat = re.compile(r"^\s*Total\s+Columns:\s*(\d+)\s*$", re.IGNORECASE)
	# tile: <id> location: [<r>, <c>] [layer: <L>]
	tile_pat_layer = re.compile(r"tile:\s*(\d+)\s+location:\s*\[(\d+)\s*,\s*(\d+)\]\s+layer:\s*(-?\d+)", re.IGNORECASE)
	tile_pat = re.compile(r"tile:\s*(\d+)\s+location:\s*\[(\d+)\s*,\s*(\d+)\]", re.IGNORECASE)

	with open(file_path, "r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			m = row_pat.match(line)
			if m:
				rows = int(m.group(1))
				continue
			m = col_pat.match(line)
			if m:
				cols = int(m.group(1))
				continue
			m = tile_pat_layer.search(line)
			if m:
				tile_id = int(m.group(1))
				r = int(m.group(2))
				c = int(m.group(3))
				layer = int(m.group(4))
				placements.append((tile_id, r, c))
				tile_layers[tile_id] = layer
				continue
			m2 = tile_pat.search(line)
			if m2:
				tile_id = int(m2.group(1))
				r = int(m2.group(2))
				c = int(m2.group(3))
				placements.append((tile_id, r, c))

	if rows is None or cols is None:
		# Fallback: infer from max observed indices if headers missing
		if placements:
			max_r = max(p[1] for p in placements)
			max_c = max(p[2] for p in placements)
			rows = (max_r + 1) if rows is None else rows
			cols = (max_c + 1) if cols is None else cols
		else:
			raise ValueError("No rows/columns information and no placements found in file.")

	return rows, cols, placements, tile_layers

def parse_placing_3d(file_path: str) -> Tuple[int, int, int, List[Tuple[int, int, int, int]], Dict[int, int]]:
	"""Parse placing.txt with support for 3D coordinates.

	Returns: (rows, cols, tiers, placements3d, tile_layers)
	  - placements3d: list of (tile_id, x, y, z)
	"""
	if not os.path.isfile(file_path):
		raise FileNotFoundError(f"placing file not found: {file_path}")

	rows = cols = None
	tiers = 1
	placements3d: List[Tuple[int, int, int, int]] = []
	tile_layers: Dict[int, int] = {}

	row_pat = re.compile(r"^\s*Total\s+Rows:\s*(\d+)\s*$", re.IGNORECASE)
	col_pat = re.compile(r"^\s*Total\s+Columns:\s*(\d+)\s*$", re.IGNORECASE)
	tier_pat = re.compile(r"^\s*Total\s+Tiers:\s*(\d+)\s*$", re.IGNORECASE)
	# 3D with layer
	tile3_layer = re.compile(r"tile:\s*(\d+)\s+location:\s*\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]\s+layer:\s*(-?\d+)", re.IGNORECASE)
	# 3D without layer
	tile3 = re.compile(r"tile:\s*(\d+)\s+location:\s*\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]", re.IGNORECASE)
	# 2D with layer
	tile2_layer = re.compile(r"tile:\s*(\d+)\s+location:\s*\[(\d+)\s*,\s*(\d+)\]\s+layer:\s*(-?\d+)", re.IGNORECASE)
	# 2D only
	tile2 = re.compile(r"tile:\s*(\d+)\s+location:\s*\[(\d+)\s*,\s*(\d+)\]", re.IGNORECASE)

	with open(file_path, "r", encoding="utf-8") as f:
		for line in f:
			s = line.strip()
			if not s:
				continue
			if (m := row_pat.match(s)):
				rows = int(m.group(1)); continue
			if (m := col_pat.match(s)):
				cols = int(m.group(1)); continue
			if (m := tier_pat.match(s)):
				tiers = int(m.group(1)); continue
			if (m := tile3_layer.search(s)):
				tid, x, y, z, lay = map(int, m.groups())
				placements3d.append((tid, x, y, z))
				tile_layers[tid] = lay
				continue
			if (m := tile3.search(s)):
				tid, x, y, z = map(int, m.groups())
				placements3d.append((tid, x, y, z))
				continue
			if (m := tile2_layer.search(s)):
				tid, x, y, lay = map(int, m.groups())
				placements3d.append((tid, x, y, 0))
				tile_layers[tid] = lay
				continue
			if (m := tile2.search(s)):
				tid, x, y = map(int, m.groups())
				placements3d.append((tid, x, y, 0))
				continue

	if rows is None or cols is None:
		if placements3d:
			max_x = max(p[1] for p in placements3d)
			max_y = max(p[2] for p in placements3d)
			rows = (max_x + 1) if rows is None else rows
			cols = (max_y + 1) if cols is None else cols
		else:
			raise ValueError("No rows/columns information and no placements found in file.")

	return rows, cols, max(tiers, 1), placements3d, tile_layers

def parse_meshconnect_for_layers(mesh_path: str, total_tiles: int) -> Dict[int, int]:
	"""Parse meshconnect.txt to build tile -> layer mapping.

	Expected lines like:
	  prelayer: X  nextlayer: Y  ... prelayer_tile: a b   nextlayer_tile: c d
	"""
	if not os.path.isfile(mesh_path):
		return {}
	tile_layers: Dict[int, int] = {}
	with open(mesh_path, 'r', encoding='utf-8') as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			# Use regex to be robust to spacing
			try:
				pre_m = re.search(r"prelayer:\s*(\d+)", line)
				next_m = re.search(r"nextlayer:\s*(\d+)", line)
				pl_m = re.search(r"prelayer_tile:\s*(\d+)\s+(\d+)", line)
				nl_m = re.search(r"nextlayer_tile:\s*(\d+)\s+(\d+)", line)
				if pre_m and pl_m:
					prelayer = int(pre_m.group(1))
					a, b = int(pl_m.group(1)), int(pl_m.group(2))
					for t in range(a, b + 1):
						if 0 <= t < total_tiles:
							tile_layers[t] = prelayer
				if next_m and nl_m:
					nextlayer = int(next_m.group(1))
					c, d = int(nl_m.group(1)), int(nl_m.group(2))
					for t in range(c, d + 1):
						if 0 <= t < total_tiles:
							tile_layers[t] = nextlayer
			except Exception:
				# Skip malformed lines
				continue
	return tile_layers


def draw_mapping(rows: int, cols: int, placements: List[Tuple[int, int, int]], save_path: Optional[str] = None) -> None:
	"""Draw the tile placement on a rows x cols grid.

	- rows, cols: grid size (0-indexed positions expected in placements)
	- placements: list of (tile_id, row, col)
	- save_path: path to save the generated image (PNG). If None, won't save.
	"""
	# Prepare canvas
	# Scale figure size relative to grid for readability
	base = 0.35
	fig_w = max(6.0, cols * base)
	fig_h = max(6.0, rows * base)
	fig, ax = plt.subplots(figsize=(fig_w, fig_h))

	# Draw grid background
	ax.set_xlim(-0.5, cols - 0.5)
	ax.set_ylim(-0.5, rows - 0.5)
	ax.set_aspect('equal')

	# Disable ticks; draw per-cell borders instead of tick-based grids
	ax.set_xticks([]); ax.set_yticks([])
	ax.tick_params(which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False, length=0)
	ax.minorticks_off()

	# Invert y-axis so row 0 is at the top (optional; comment out to have origin at bottom-left)
	ax.invert_yaxis()

	# Plot tiles: color each cell by tile id and label numbers
	if placements:
		# Determine colormap normalization
		tile_ids = [p[0] for p in placements]
		norm = plt.Normalize(vmin=min(tile_ids), vmax=max(tile_ids))
		cmap = cm.viridis

		def text_color_for_rgb(rgba):
			# Choose black/white for contrast based on luminance
			r, g, b, _ = rgba
			luminance = 0.299 * r + 0.587 * g + 0.114 * b
			return 'black' if luminance > 0.6 else 'white'

		# Draw colored cells
		for tid, r, c in placements:
			color = cmap(norm(tid))
			rect = patches.Rectangle((c - 0.5, r - 0.5), 1.0, 1.0, facecolor=color, edgecolor='none', zorder=1, linewidth=0)
			ax.add_patch(rect)
			ax.text(c, r, str(tid), fontsize=7, ha='center', va='center', color=text_color_for_rgb(color), zorder=4)

		# Draw per-cell borders on top to emulate gridlines
		for r in range(rows):
			for c in range(cols):
				rect_border = patches.Rectangle((c - 0.5, r - 0.5), 1.0, 1.0,
											   facecolor='none', edgecolor='#e0e0e0',
											   linewidth=0.8, zorder=2)
				ax.add_patch(rect_border)

	ax.set_title(f"Tile Mapping ({rows} x {cols})", fontsize=12)

	plt.tight_layout()
	if save_path:
		out_dir = os.path.dirname(save_path)
		if out_dir and not os.path.isdir(out_dir):
			os.makedirs(out_dir, exist_ok=True)
		fig.savefig(save_path, dpi=200)
	plt.close(fig)

def draw_layer_map(rows: int, cols: int, placements: List[Tuple[int, int, int]], tile_layers: Dict[int, int], save_path: Optional[str] = None) -> None:
	"""Draw a grid labeling each tile with its layer number.
	"""
	if not placements:
		return
	if not tile_layers:
		return

	# Prepare canvas
	base = 0.35
	fig_w = max(6.0, cols * base)
	fig_h = max(6.0, rows * base)
	fig, ax = plt.subplots(figsize=(fig_w, fig_h))

	ax.set_xlim(-0.5, cols - 0.5)
	ax.set_ylim(-0.5, rows - 0.5)
	ax.set_aspect('equal')

	# Disable ticks; draw per-cell borders instead of tick-based grids
	ax.set_xticks([]); ax.set_yticks([])
	ax.tick_params(which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False, length=0)
	ax.minorticks_off()
	ax.invert_yaxis()

	# Determine unique layers for colormap
	layers = sorted({layer for layer in tile_layers.values()})
	# Avoid -1 dominance: if only -1 exists, keep a neutral cmap
	vmin = min(layers) if layers else 0
	vmax = max(layers) if layers else 1
	norm = plt.Normalize(vmin=vmin, vmax=vmax)
	cmap = cm.tab20 if len(layers) <= 20 else cm.nipy_spectral

	def text_color_for_rgb(rgba):
		r, g, b, _ = rgba
		luminance = 0.299 * r + 0.587 * g + 0.114 * b
		return 'black' if luminance > 0.6 else 'white'

	for tid, r, c in placements:
		layer = tile_layers.get(tid, -1)
		color = cmap(norm(layer)) if layer != -1 else (0.9, 0.9, 0.9, 1.0)
		rect = patches.Rectangle((c - 0.5, r - 0.5), 1.0, 1.0, facecolor=color, edgecolor='none', zorder=1, linewidth=0)
		ax.add_patch(rect)
		ax.text(c, r, str(layer), fontsize=7, ha='center', va='center', color=text_color_for_rgb(color), zorder=4)

	# Draw per-cell borders on top to emulate gridlines
	for r in range(rows):
		for c in range(cols):
			rect_border = patches.Rectangle((c - 0.5, r - 0.5), 1.0, 1.0,
										   facecolor='none', edgecolor='#e0e0e0',
										   linewidth=0.8, zorder=2)
			ax.add_patch(rect_border)

	ax.set_title(f"Layer Map ({rows} x {cols})", fontsize=12)
	plt.tight_layout()
	if save_path:
		out_dir = os.path.dirname(save_path)
		if out_dir and not os.path.isdir(out_dir):
			os.makedirs(out_dir, exist_ok=True)
		fig.savefig(save_path, dpi=200)
	plt.close(fig)

def draw_mapping_subplots(rows: int, cols: int, tiers: int,
						  placements3d: List[Tuple[int, int, int, int]],
						  save_path: Optional[str] = None) -> None:
	"""Draw a single figure with tiers subplots, one per tier, showing mapping by tile id."""
	if not placements3d or tiers <= 0:
		return

	base = 0.35
	fig_w = max(9.0, tiers * cols * base)
	fig_h = max(5.0, rows * base)
	fig, axes = plt.subplots(1, tiers, figsize=(fig_w, fig_h), squeeze=False)
	axes = axes[0]

	# Colormap by tile id
	tile_ids = [tid for (tid, *_rest) in placements3d]
	norm = plt.Normalize(vmin=min(tile_ids), vmax=max(tile_ids))
	cmap = cm.viridis

	def text_color_for_rgb(rgba):
		r, g, b, _ = rgba
		luminance = 0.299 * r + 0.587 * g + 0.114 * b
		return 'black' if luminance > 0.6 else 'white'

	for z in range(tiers):
		ax = axes[z]
		ax.set_xlim(-0.5, cols - 0.5)
		ax.set_ylim(-0.5, rows - 0.5)
		ax.set_aspect('equal')
		ax.set_xticks([]); ax.set_yticks([])
		ax.tick_params(which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False, length=0)
		# Keep all four spines (outer border)
		for spine in ax.spines.values():
			spine.set_visible(True)
			spine.set_linewidth(1.0)
			spine.set_color('black')
		ax.invert_yaxis()
		
		# Fill tiles for this tier
		for tid, r, c, zz in placements3d:
			if zz != z: continue
			color = cmap(norm(tid))
			rect = patches.Rectangle((c - 0.5, r - 0.5), 1.0, 1.0, facecolor=color, edgecolor='none', zorder=1)
			ax.add_patch(rect)
			ax.text(c, r, str(tid), fontsize=7, ha='center', va='center', color=text_color_for_rgb(color), zorder=4)
		# Draw cell borders
		for r in range(rows):
			for c in range(cols):
				rect_border = patches.Rectangle((c - 0.5, r - 0.5), 1.0, 1.0, facecolor='none', edgecolor='#DDDDDD', linewidth=0.8, zorder=2)
				ax.add_patch(rect_border)

		ax.set_title(f"Tier {z}")

	plt.tight_layout()
	if save_path:
		out_dir = os.path.dirname(save_path)
		if out_dir and not os.path.isdir(out_dir):
			os.makedirs(out_dir, exist_ok=True)
		fig.savefig(save_path, dpi=200)
	plt.close(fig)

def draw_layer_map_subplots(rows: int, cols: int, tiers: int,
							placements3d: List[Tuple[int, int, int, int]],
							tile_layers: Dict[int, int],
							save_path: Optional[str] = None) -> None:
	"""Draw a single figure with tiers subplots, one per tier, colored and labeled by layer id."""
	if not placements3d or not tile_layers:
		return

	base = 0.35
	fig_w = max(9.0, tiers * cols * base)
	fig_h = max(5.0, rows * base)
	fig, axes = plt.subplots(1, tiers, figsize=(fig_w, fig_h), squeeze=False)
	axes = axes[0]

	layers = sorted({lay for lay in tile_layers.values()})
	vmin = min(layers) if layers else 0
	vmax = max(layers) if layers else 1
	norm = plt.Normalize(vmin=vmin, vmax=vmax)
	cmap = cm.tab20 if len(layers) <= 20 else cm.nipy_spectral

	def text_color_for_rgb(rgba):
		r, g, b, _ = rgba
		luminance = 0.299 * r + 0.587 * g + 0.114 * b
		return 'black' if luminance > 0.6 else 'white'

	for z in range(tiers):
		ax = axes[z]
		ax.set_xlim(-0.5, cols - 0.5)
		ax.set_ylim(-0.5, rows - 0.5)
		ax.set_aspect('equal')
		ax.set_xticks([]); ax.set_yticks([])
		ax.tick_params(which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False, length=0)
		ax.minorticks_off()
		ax.invert_yaxis()
		
		# Fill for this tier
		for tid, r, c, zz in placements3d:
			if zz != z: continue
			lay = tile_layers.get(tid, -1)
			color = cmap(norm(lay)) if lay != -1 else (0.9, 0.9, 0.9, 1.0)
			rect = patches.Rectangle((c - 0.5, r - 0.5), 1.0, 1.0, facecolor=color, edgecolor='none', zorder=1)
			ax.add_patch(rect)
			ax.text(c, r, str(lay), fontsize=8, ha='center', va='center', color=text_color_for_rgb(color), zorder=4)
		# Cell borders
		for r in range(rows):
			for c in range(cols):
				rect_border = patches.Rectangle((c - 0.5, r - 0.5), 1.0, 1.0, facecolor='none', edgecolor='#DDDDDD', linewidth=0.8, zorder=2)
				ax.add_patch(rect_border)

		ax.set_title(f"Tier {z}")

	# Shared legend on the right
	import matplotlib.patches as mpatches
	handles = []
	for lay in layers:
		color = cmap(norm(lay)) if lay != -1 else (0.9, 0.9, 0.9, 1.0)
		label = f"Layer {lay}" if lay != -1 else "Layer -1 (unknown)"
		handles.append(mpatches.Patch(color=color, label=label))
	if handles:
		leg = fig.legend(handles=handles, loc='center left', bbox_to_anchor=(1.01, 0.5), borderaxespad=0.)
		leg.set_frame_on(True)
		leg.get_frame().set_facecolor('none')
		leg.get_frame().set_alpha(0.0)

	plt.tight_layout()
	if save_path:
		out_dir = os.path.dirname(save_path)
		if out_dir and not os.path.isdir(out_dir):
			os.makedirs(out_dir, exist_ok=True)
		fig.savefig(save_path, dpi=200, bbox_inches='tight')
	plt.close(fig)

def draw_layer_map_3d_stacked(rows: int, cols: int, tiers: int,
							  placements3d: List[Tuple[int, int, int, int]],
							  tile_layers: Dict[int, int],
							  save_path: Optional[str] = None,
							  title: str = "3D Layer Map") -> None:
	"""Draw a single 3D figure with all tiers stacked. Cells are colored by layer id."""
	if not placements3d or not tile_layers:
		return

	fig = plt.figure(figsize=(max(7.0, cols * 0.7), max(6.5, rows * 0.7)))
	ax = fig.add_subplot(111, projection='3d')

	# Colormap based on layer values
	layers = sorted({lay for lay in tile_layers.values()})
	vmin = min(layers) if layers else 0
	vmax = max(layers) if layers else 1
	norm = plt.Normalize(vmin=vmin, vmax=vmax)
	cmap = cm.tab20 if len(layers) <= 20 else cm.nipy_spectral

	# Increase spacing between tiers: much larger spacing to avoid occlusion
	z_step = max(rows, cols) * 5

	# Draw faint full-layer grid outlines for context
	outline_faces = []
	outline_colors = []
	for z in range(max(tiers, 1)):
		z_plot = z * z_step
		for r in range(rows):
			for c in range(cols):
				verts = [
					(c - 0.5, r - 0.5, z_plot),
					(c + 0.5, r - 0.5, z_plot),
					(c + 0.5, r + 0.5, z_plot),
					(c - 0.5, r + 0.5, z_plot),
				]
				outline_faces.append(verts)
				outline_colors.append((1, 1, 1, 1))
	if outline_faces:
		poly_outline = Poly3DCollection(outline_faces, facecolors=outline_colors, edgecolors='k', linewidths=0.7, alpha=1.0)
		ax.add_collection3d(poly_outline)

	faces = []
	colors = []
	# To build legend for present layers only
	present_layers = set()
	for tid, r, c, z in placements3d:
		lay = tile_layers.get(tid, -1)
		color = cmap(norm(lay)) if lay != -1 else (0.9, 0.9, 0.9, 1.0)
		z_plot = z * z_step
		verts = [
			(c - 0.5, r - 0.5, z_plot),
			(c + 0.5, r - 0.5, z_plot),
			(c + 0.5, r + 0.5, z_plot),
			(c - 0.5, r + 0.5, z_plot),
		]
		faces.append(verts)
		colors.append(color)
		present_layers.add(lay)

	poly = Poly3DCollection(faces, facecolors=colors, edgecolors='white', linewidths=0.2, alpha=0.85)
	ax.add_collection3d(poly)

	# Annotate each placed tile with its layer number
	for tid, r, c, z in placements3d:
		lay = tile_layers.get(tid, -1)
		z_plot = z * z_step
		txt = ax.text(c, r, z_plot + 0.01, str(lay), ha='center', va='center', fontsize=7, color='black', zorder=10)
		txt.set_path_effects([patheffects.withStroke(linewidth=1.5, foreground='white')])

	ax.set_xlim(-0.5, cols - 0.5)
	ax.set_ylim(-0.5, rows - 0.5)
	ax.set_zlim(-0.5 * z_step, (tiers - 0.5) * z_step)
	# Hide axes and labels as requested
	ax.set_axis_off()
	# Adjust view to reduce overlap
	ax.view_init(elev=30, azim=-45)
	ax.set_title(title)

	# Legend on the side with present layers
	import matplotlib.patches as mpatches
	handles = []
	for lay in sorted(present_layers):
		color = cmap(norm(lay)) if lay != -1 else (0.9, 0.9, 0.9, 1.0)
		label = f"Layer {lay}" if lay != -1 else "Layer -1 (unknown)"
		handles.append(mpatches.Patch(color=color, label=label))
	if handles:
		ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)

	plt.tight_layout()
	if save_path:
		out_dir = os.path.dirname(save_path)
		if out_dir and not os.path.isdir(out_dir):
			os.makedirs(out_dir, exist_ok=True)
		fig.savefig(save_path, dpi=200)
	plt.close(fig)


def main():
	parser = argparse.ArgumentParser(description="Visualize placing.txt mapping result")
	parser.add_argument("--input", "-i", type=str, default=os.path.join(os.path.dirname(__file__), "..", "Parameters", "placing.txt"), help="Path to placing.txt")
	parser.add_argument("--output", "-o", type=str, default=None, help="Path to save the output image (PNG)")
	parser.add_argument("--tier", type=int, default=None, help="Tier index to visualize (0-based). If omitted and --all-tiers not set, defaults to 0 for 3D.")
	parser.add_argument("--all-tiers", action="store_true", help="Render all tiers as separate images when 3D is present.")
	args = parser.parse_args()

	rows, cols, tiers, placements3d, tile_layers = parse_placing_3d(os.path.abspath(args.input))

	# Default output path: next to input
	save_path = args.output
	if save_path is None:
		base_dir = os.path.dirname(os.path.abspath(args.input))
		save_path = os.path.join(base_dir, "placing_map.png")

	def per_tier_placements(z: int) -> List[Tuple[int, int, int]]:
		return [(tid, x, y) for (tid, x, y, zz) in placements3d if zz == z]

	def total_tiles_count() -> int:
		return (max((tid for (tid, *_rest) in placements3d), default=-1) + 1)

	if tiers <= 1:
		# 2D case: use z=0 slice implicitly
		placements2d = per_tier_placements(0) if placements3d else []
		draw_mapping(rows, cols, placements2d, save_path=save_path)
		# Layer figure
		layer_save = None
		if save_path:
			root, ext = os.path.splitext(save_path)
			layer_save = root.replace("placing_map", "placing_layer_map") + ext
		else:
			layer_save = None
		if not tile_layers:
			base_dir = os.path.dirname(os.path.abspath(args.input))
			mesh_path = os.path.join(base_dir, "meshconnect.txt")
			tile_layers = parse_meshconnect_for_layers(mesh_path, total_tiles_count())
		if tile_layers:
			draw_layer_map(rows, cols, placements2d, tile_layers, save_path=layer_save)
			if layer_save: print(f"Saved layer visualization to: {layer_save}")
		else:
			print("No layer information found in placing.txt or meshconnect.txt; skipped layer plot.")
		print(f"Saved mapping visualization to: {save_path}")
		return
	else:
		# 3D case: render a single figure with tiers subplots (no separate images)
		if save_path:
			root, ext = os.path.splitext(save_path)
			map_sub = f"{root.replace('placing_map', 'placing_map_subplots')}{ext}"
			layer_sub = f"{root.replace('placing_map', 'placing_layer_map_subplots')}{ext}"
		else:
			base_dir = os.path.dirname(os.path.abspath(args.input))
			map_sub = os.path.join(base_dir, 'placing_map_subplots.png')
			layer_sub = os.path.join(base_dir, 'placing_layer_map_subplots.png')

		draw_mapping_subplots(rows, cols, tiers, placements3d, save_path=map_sub)
		print(f"Saved 3D mapping visualization (subplots) to: {map_sub}")

		if not tile_layers:
			base_dir = os.path.dirname(os.path.abspath(args.input))
			mesh_path = os.path.join(base_dir, "meshconnect.txt")
			tile_layers = parse_meshconnect_for_layers(mesh_path, total_tiles_count())
		if tile_layers:
			draw_layer_map_subplots(rows, cols, tiers, placements3d, tile_layers, save_path=layer_sub)
			print(f"Saved 3D layer visualization (subplots) to: {layer_sub}")
		else:
			print("No layer information found in placing.txt or meshconnect.txt; skipped layer subplot plot.")

		# Additionally, generate a single stacked 3D figure for mapping and layer
		if placements3d:
			root, ext = os.path.splitext(save_path) if save_path else (os.path.join(os.path.dirname(os.path.abspath(args.input)), 'placing_map'), '.png')
			# Layer stacked
			if not tile_layers:
				base_dir = os.path.dirname(os.path.abspath(args.input))
				mesh_path = os.path.join(base_dir, "meshconnect.txt")
				tile_layers = parse_meshconnect_for_layers(mesh_path, total_tiles_count())
			if tile_layers:
				stack_layer = f"{root.replace('placing_map', 'placing_layer_map_3d')}{ext}"
				draw_layer_map_3d_stacked(rows, cols, tiers, placements3d, tile_layers, save_path=stack_layer)
				print(f"Saved 3D layer visualization to: {stack_layer}")


if __name__ == "__main__":
	main()

