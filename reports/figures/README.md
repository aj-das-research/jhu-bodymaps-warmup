# Diagnostic figures

Static comparison panels for the BodyMaps warm-up (not committed as binaries).

## Layout

```text
reports/figures/
  raw_vs_shapekit/
    <case_id>/                e.g. BDMAP_00000006
      <vertebra>/             e.g. vertebrae_C1
        <axis>_<slice>_panel.png
```

## Examples

Single slice (defaults: case 6, C1, axial 319):

```bash
python scripts/plot_slice_compare_panel.py
# -> reports/figures/raw_vs_shapekit/BDMAP_00000006/vertebrae_C1/axial_319_panel.png
```

Every axial plane where raw ≠ ShapeKit:

```bash
python scripts/plot_slice_compare_panel.py --diff_slices
```

PNG/PDF outputs are gitignored; this README is tracked so the tree stays documented.
