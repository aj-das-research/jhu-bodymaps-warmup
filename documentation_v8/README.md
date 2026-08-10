# documentation_v8: split-nodule and fused-blade correction study

Goal (from the annotated review): a facet knob or a spinous blade is one
rigid anatomical piece and must carry ONE label; a label boundary may only
pass through joint clefts, PV plates, and thin necks, never through the
interior of a thick mass; and consecutive-slice (temporal) plus multiview
physics should decide identity where any single 3D view fails.

## Defect quantification (case BDMAP_00000031, on the v7 output)

Two meters were built (scripts/diag_v8.py and in-pipeline):

1. MASSCUT: area of label boundary crossing bone thicker than 2.5 mm.
   v7 measured 10-39% of every adjacent-pair interface, max 5-6 mm -
   boundaries through solid pieces (figures/01, red = violations).
   Caveat learned later: the correct boundary through a PV-FUSED joint is
   legitimately thick, so masscut is reported, not gated on.
2. BADCUT PIECES (the gate): supra-neck in-plane 2D core pieces whose
   INTERNAL label boundary crosses bone >= 2.6 mm. A mixed piece whose
   boundary sits at a thin waist is a legitimate joint through a fused
   pair and does not count. v7: 47 pieces (21/21 parasagittal slices).

## Methods tried, in order, with measured outcomes

All variants ran gated (defect meter must improve, audits must not
degrade, per-level shift <= 4 cm3; else auto-revert). Failures below were
caught by the gates, not by luck.

A. FIXED-THRESHOLD SKELETON DECOMPOSITION (bone EDT >= 1.6 mm components,
   identity = contained body core). FAILED: this pathology fuses three
   levels thicker than any safe threshold - the whole spine was ONE
   452 cm3 component; tracking inside it scattered labels
   (masscut 4k -> 54k mm2). Auto-reverted.
B. 3D SADDLE RACE (watershed on -EDT from body cores). Halved masscut
   (3977 -> 2122 mm2) but swapped whole arches through fused discs
   (T10 -23.6 cm3, T9 +26.5 cm3): through a 4-5 mm fused disc a
   neighbor's front outruns the owner's own lamina path. Auto-reverted.
C. PER-SLICE UNIFORM RACE + TEMPORAL CHAINS (in-plane race from body
   pixels; IoU chains). IoU breaks at split/merge events (blade merging
   into lamina between slices), containment-matching percolates chains
   across levels, first-come propagation locks wrong labels, and
   label-set reachability still leaked through one-way conduits (fused
   facet pillars reachable by exactly one label: T7 +9.6 cm3).
   All variants auto-reverted. Lesson: global re-derivation - 3D or
   chained 2D - cannot be made safe on ankylosed anatomy.
D. PER-PIECE SURGERY (accepted; now stage 2f of the pipeline):
   - A mixed in-plane core piece IS the defect by definition (one rigid
     cross-section piece carrying two labels).
   - UNIFY to the majority label iff the internal boundary crosses the
     piece's thick interior (>= 1.8 mm) and the minority is <= 40%.
   - If the piece cannot be unified (near-50/50: genuinely fused pair or
     ankylosed multi-level chain - measured pieces contain up to FIVE
     levels at ~25% each), RELOCATE the internal boundary to the piece's
     in-plane thickness valley (watershed on -EDT2D seeded from each
     label's eroded end), accepted only if the boundary gets strictly
     thinner.
   - Runs over all three orthogonal views in the shear-straightened
     frame; recolor-only (raw envelope preserved); seams converged with
     the volume-preserving interface majority vote; orphan slivers fused
     (nothing deleted). One pass is the fixed point (a second pass
     measurably degrades and self-reverts).

Outcome on case 31 (pre-final run): badcut pieces 47 -> 36, changed
~24-26 cm3 with every per-level shift under 2.1 cm3, audits clean.
Residual badcut pieces are inside ankylosed multi-level fusions where a
boundary must cross fused bone somewhere; the relocation places it at the
thinnest available surface. figures/02 shows the knob region T6-T10
posterior, v7 vs v8: knobs increasingly wrap to their own level.

## Standing physics lessons recorded in the deliverable docstrings

- Plate-on-plate contacts are thick to 3D EDT but thin LINES in the
  cross-section of the view containing the structure's long axis: measure
  and cut in-plane (multiview), not in 3D.
- Fixed viewing planes break on curved spines: shear-straighten with the
  column's own centerline first (integer per-slice shifts, exactly
  invertible).
- The view that sees a structure lengthwise holds its connectivity truth
  (elongation-gated voting, stage 2e).
- Trust flows from the disc-cut-validated BODIES; label-path anchors are
  circular in contested territory.
- Every stage carries its own defect meter and reverts itself; on this
  pathology the meters vetoed four plausible algorithms before one passed.

## Files

- figures/01: v7 boundaries crossing >= 2.5 mm bone (red), posterior.
- figures/02: knob region posterior, v7 vs v8 candidate.
- figures/03: why fixed-plane multiview abstained (chained component).
- figures/04: v7 midsagittal blade panels (raw vs refined).
- figures/05: pedicle-root arch phantom (v5 foundation).
- figures/06: raw-labeled bone dropped by early versions, reclaimed in v6.
- Pipeline QA for the v8 run: reports/ *_postprocessing_qa.json ("skeleton"
  block = stage 2f record).
- v8 volumes: AbdomenAtlasDemoPredict_refined/<case>/ (also saved as
  combined_labels_v8.nii.gz next to each combined_labels.nii.gz).
