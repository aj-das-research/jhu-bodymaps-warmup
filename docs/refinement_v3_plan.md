# v3 plan: evidence-local boundaries that interdigitate like anatomy

Case study: BDMAP_00000031/06 (SuPreM vertebrae warm-up). Status: v2 fixed level
OWNERSHIP (L1 restored 13.9->65.8 cm3, all audits clean both cases) but its
interfaces are near-planar guillotines that amputate articular processes.

## Measured diagnosis (2026-08-09)

Interface planarity = RMS residual (mm) of interface voxels vs their best-fit
plane. Anatomy interdigitates (facets wrap, spinous spans levels), so a CORRECT
interface is far from planar. Raw prediction vs our v2, case 31:

| pair | raw RMS | v2 RMS | | pair | raw RMS | v2 RMS |
|---|---|---|---|---|---|---|
| T3\|T4 | 6.96 | 2.40 | | T7\|T8 | 7.15 | 0.81 |
| T4\|T5 | 4.02 | 0.37 | | T8\|T9 | 6.53 | 3.47 |
| T6\|T7 | 7.95 | 1.63 | | T9\|T10 | 6.83 | 0.97 |

v2 flattened interfaces by 4-8x. Root cause, precisely: the uniform-BFS
competition places boundaries at EQUAL GEODESIC DISTANCE from body seeds, and
the s-cut domain gating imprints planes - neither consults the image at the
boundary itself. The low-HU clefts (facet joint spaces, interspinous ligament,
disc) are exactly where boundaries belong, and only local image evidence can
put them there. Euler-characteristic and provenance findings from the v2 QC
round stand (arch river, tilted cuts) - v3 must keep those wins.

## Design principles

1. Global machinery (disc-profile segments, DP cuts, suspicion bands) decides
   ONLY level identity - which vertebra a region belongs to. It must never
   shape the local surface.
2. The local interface is decided by LOCAL image evidence: boundaries settle
   where HU is weakest between two claims (joint space physics: cortical shell
   - cartilage gap - cortical shell reads as a bright-dark-bright profile).
3. Interdigitation is expected, not an error: per-level shapes inspected in
   isolation must show body, pedicles, laminae, both articular process pairs,
   transverse processes, spinous process. z-overlap of neighbors at facets
   (roughly 8-15 mm in T/L) is the healthy range; ~0 means guillotine, ~25 mm
   means bleed.
4. Every stage emits debug artifacts before it is accepted: overlay PNGs
   (sagittal + coronal), isolated per-vertebra 4-view sheets, metric JSON with
   deltas. reports/debug/<case>/<stage>/ is part of the deliverable.

## Literature and code to build on

- Payer et al., VISAPP 2020 (VerSe winner): coarse-to-fine, then EACH vertebra
  segmented in its OWN crop - per-instance masks may overlap and are resolved
  afterward; validates principle 3. Code: github.com/christianpayer.
- Lessmann et al., MedIA 2019 (arXiv:1804.04383): iterative instance memory,
  sliding through the column - the "temporal" slice-tracking intuition:
  consecutive cross-sections carry boundary evidence a single slice lacks.
- Sekuboyina et al., VerSe benchmark (arXiv:2001.09193): error taxonomy
  (off-by-one identity vs surface errors) - our stage split mirrors it.
- Grady, TPAMI 2006 random walker: multilabel seeded segmentation whose
  boundaries settle on weak image edges via conductance exp(-beta*|dHU|^2);
  exists in scikit-image. Extension with label priors: Grady ICCV05.
- Graph cuts / alpha-expansion (PyMaxflow) as the discrete alternative:
  Potts pairwise weighted by edge strength -> minimal-cut surfaces along
  clefts.
- Morphological active contours (ACWE/GAC, in scikit-image): per-vertebra
  surface refinement with image forces, bounded evolution.

## Staged plan (each stage gated by metrics + rendered evidence)

P1 - edge-aware competition (replaces uniform BFS inside the band).
  Experiment order, cheapest first, same seeds/segments as v2:
  a. watershed(image=|grad HU|) - fronts stall at cortical cliffs and meet in
     the cleft valleys; 5-line change.
  b. random_walker(beta tuned; solved on the band slab, 2x downsampled, then
     upsampled and locally re-solved at full res near boundaries).
  c. alpha-expansion graph cut if (a)/(b) underperform.
  Gate: T-band planarity RMS rises to >= 4 mm (raw-like) while chi stays
  <= v2, volumes stay smooth (logres < 0.30), audits stay clean, and case 6
  stays a no-op outside its clean-band behavior.
  Debug: per-pair interface renders + planarity/chi/volume delta table.

P2 - 2.5D temporal consistency (the slice-tracking idea, Lessmann-style).
  STATUS: DEFERRED / conditional (decision 2026-08-09) - execute only if the
  P1 gate is not met by P1b (surface un-gating + random walker). If invoked,
  ship the detector before any corrector.
  Parameterize the column by arc length s. Per level, track the 2D
  cross-section along s: area A(s), centroid c(s), contour IoU(s, s+1).
  Handover artifacts appear as discontinuities (IoU cliff, area jump) exactly
  at cut planes. First ship as a DETECTOR (flags + plots per level); then as
  a corrector: re-solve flagged shells locally with +/-k-slice context
  (contour propagation with smoothness prior; Kalman-like update).
  Debug: A(s)/IoU(s) curves per level with flags; before/after overlays.

P3 - per-vertebra isolated refinement (Payer-style crops).
  For each level: padded crop, morphological active contour initialized from
  the current mask, image forces from CT, neighbors as soft exclusion (not
  hard walls), deviation cap ~2 mm, volume-preservation tolerance. This is
  what regrows amputated articular processes without global re-flooding.
  Gate: process-completeness score up, Dice vs raw in uncontested zones ~1.
  Debug: isolated 4-view sheet per level, before/after.

P4 - shape-completeness audit (extends stage 4).
  Per T/L level: detect 4 articular processes + 2 TP + 1 SP as protrusions of
  the arch skeleton; interdigitation in [8, 15] mm at facet pairs; planarity
  in [4, 9] mm; chi in [-4, 2]; left/right symmetry check. Flags only.

P5 - transfer + guards. Identical parameters on both cases; strict-improvement
  revert per band retained; runtime budget <= 15 min/case CPU. Then the same
  audit on any new case is the generalization test.

## Non-goals for the warm-up

Statistical shape models / template registration (Klinder 2009) and learned
per-vertebra U-Nets are the "next model" answer, not post-processing; the
report should cite them as the ceiling and position v3 as the classical
evidence-based approximation of it.
