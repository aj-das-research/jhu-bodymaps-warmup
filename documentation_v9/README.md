# documentation_v9: the spinous one-down chain - diagnosis and repair

High-resolution review of the v8 output (zoomed ITK-SNAP 3D) found spinous
processes assigned one level DOWN through T9..L2 on BDMAP_00000031:
T9's blade wore T10's color, T10's wore T11's, T11's wore T12's, T12's wore
L1's, and L1's blade wore L2's - so L2 appeared to own two spinous
processes. The v8 gallery renders had downsampled labels to ~1.4 mm for
rotation speed and MASKED exactly this defect; all verification renders in
this study are now full resolution (scripts/diag_blades.py).

## 1. What the data showed at full resolution

- RAW is not "correct minus one shift": raw blades are messy MIXTURES, with
  each label also overreaching DOWNWARD onto the next blade's root
  (visualizations/debug_spinous/C_midsag_slab_fullres.png, left panel).
- v8 is CLEANLY wrong: sharp one-down assignment, T9's whole blade blue
  (right panel; E_violation_overlay.png shows the impossible volumes in red).
- The blade ROOT-STRIP audit (which lamina junction each label's corridor
  mass attaches to) reads: raw = 7 levels shifted; v8 = all junctions OK
  but tips stolen - the steal spared the strip, so only a full-blade view
  catches it.

## 2. The algorithmic mistake (three compounding parts)

1. STAGE 2B'S ARCH RACE DECIDES IDENTITY BY ARRIVAL ORDER. Its waist-severed
   core race severs bone thinner than 1.5 mm (facet necks) before racing.
   On this DISH/ankylosed spine the OSSIFIED INTERSPINOUS BRIDGES are
   thicker than that, so the identity race leaked across them - and a
   steeply imbricated blade's tip is Euclidean-NEARER the vertebra below
   (that is what imbrication means), so every leak handed the drooping half
   one level down. QA evidence: the T5..L2 band was accepted on its audit
   gate (badness 5 -> 1, bodies genuinely fixed) while silently minting
   3.16 cm3 of upward-violating blade volume - no meter watched blades.
2. NO DOWNSTREAM STAGE COULD RESCUE IT. Stage 2e abstained exactly here
   (fused chains = multi-anchor blocker components; MULTIVIEW_REVERTED_ALL),
   and stage 2f only sees MIXED pieces: a wholly-stolen blade is a PURE
   piece and invisible, majority-unification points the wrong way at
   junction pieces, and valley relocation moves boundaries, never identity.
   Net effect: raw's messy downward overreach was CONVERTED into a clean
   one-down assignment.
3. VERIFICATION RENDERED TOO COARSE. The 1.4 mm gallery hid a defect that
   is obvious at native 0.9 x 0.9 x 0.7 mm. (Fixed: full-res debug set.)

## 3. The meter: spinous UPWARD-VIOLATION (upv)

In the midline posterior corridor the only structures are spinous blades
and interspinous bone, and blades angle strictly CAUDALLY: label mass above
its own body-band top is anatomically impossible - it is a blade wearing
the label of the level below. One-sided by design (the drooped part of a
stolen blade z-overlaps the thief's own band and only root-attachment can
judge it), chain-proof, object-free.

    raw 0.02 | v8 3.16 | v9 0.21 cm3   (BDMAP_00000031, T7..L3 window)

## 4. Methods tried, in order, with measured outcomes

A. MULTI-SCALE IN-PLANE PEEL RACE ("hier" arch mode, kept as an ablation
   branch): sever at ALL scales at once - claim thick bone first, fronts
   cross thinner bone later, thickness = per-parasagittal-slice 2D EDT.
   MEASURED WORSE: band upv 0.33 -> 7.68 cm3 (vs 3.16 for "core"),
   auto-reverted. WHY: the ossified interspinous bridge is a MIDLINE
   SAGITTAL SHEET, bone-continuous with both blades and BROAD in its own
   cross-section - in-plane thickness is large at exactly the bridge to be
   severed, so the peel welds blade to blade. No geometric thickness signal
   separates a blade from a fusion sheet in its own plane. (3D EDT agrees;
   this closes the geometric-signal route entirely.)
B. CAUDAL-FLOW REPAIR (stage 2g, ACCEPTED), built on the one invariant that
   survives ankylosis: NOTHING in the midline posterior corridor grows
   toward the head. The ring STRIP (6 mm at the corridor's anterior edge,
   where junctions live) seeds identity where its labels are BAND-
   CONSISTENT; everything posterior is re-derived in one top-to-bottom
   sweep over consecutive axial slices - the z+1 assignment flows down
   wherever bone continues, extended in-plane by a uniform 2D watershed. A
   label can only enter at its junction and flow caudally to the tip;
   descending fronts of consecutive levels meet inside the interspinous
   sheet. (This is the consecutive-slice "temporal outline propagation"
   idea made safe by an anatomical direction constraint.)
   Iterations, each caught by meters:
   - v1 body-centered slab: upv 3.15 -> 1.41 but the slab CLIPPED rotated
     blades (scoliosis swings the spinous line off the body centroid);
     badcut 35 -> 36 from slab-edge seams. REVERTED by gates.
   - v2 blade-tracking slab (follow the corridor's own bone centroid):
     upv -> 1.15; residual steals sat INSIDE the kept strip. REVERTED.
   - v3 band-consistent strip seeding (a mislabeled strip patch is flow
     territory, not a seed): upv -> 0.22, but five levels FRAGMENTED into
     leftover-sliver components (T10 -> 44 comps). REVERTED by audit gate.
   - v4 + convergence tail (volume-preserving interface majority vote +
     orphan-sliver fusion inside the changed-region bbox, the stage-2f
     treatment): upv 3.15 -> 0.21, badcut 35 -> 34, audit 0 -> 0,
     max per-level shift 4.5 cm3 (T9 regaining its blade). ACCEPTED.

## 5. Results (BDMAP_00000031, full pipeline from raw)

- upward-violation: 3.15 -> 0.21 cm3 (T10 1.86 -> 0.07, T11 0.83 -> 0.08,
  L1 0.46 -> 0.07); root-strip audit: no shifted levels (raw had 7).
- The five reported shifts are all resolved at full resolution: each blade
  carries its root's label through the droop; L2 owns one spinous process
  (visualizations/debug_spinous/final_v9/C_midsag_slab_fullres.png).
- Volume handbacks: T9 +4.5, T12 +1.3 cm3; L2 -1.5, L1 -2.1, T11 -1.0,
  T10 -1.2 cm3 (the chain unwinding).
- Collateral meters: badcut pieces 47 -> 31 (2f) -> 30/34-frame (2g);
  audits all clean; envelope preserved (recolor-only, nothing deleted).
- BDMAP_00000006 (clean case): stage 2g attempted an 8.6 cm3 move at L1
  (its upv "signal" there is band-edge noise at L5/T1/C7, not a real blade
  chain) and SELF-REVERTED on the over-shift gate - same protective
  pattern as stages 2e/2f on this case; output audits all-zero.

## 6. Figures (visualizations/, all full resolution unless noted)

- debug_spinous/A_posterior_fullres.png, B_oblique_fullres.png: raw vs v8
  posterior/oblique zooms - defect confirmation.
- debug_spinous/C_midsag_slab_fullres.png: raw vs v8 midsagittal slab with
  body bands and measured blade roots - the one-down chain visible.
- debug_spinous/D_parasag_slices.png: raw vs v8 parasagittal label slices.
- debug_spinous/E_violation_overlay.png: upward-violation volumes in RED.
- debug_spinous/after2g/*: v8 vs the accepted stage-2g candidate.
- debug_spinous/final_v9/*: RAW vs final v9 (C-figure = the fix, E-figure
  near-empty red).
- debug_spinous/{,post2b/,after2g/,final_v9/}blade_root_table.json: the
  measured tables behind every claim above.
- BDMAP_*_column_5views.png, BDMAP_*_standalone_sheet.png: five-angle and
  per-vertebra galleries, raw vs v9 (overview only: ~1.4 mm; use the
  debug_spinous set for fine verification).

## 7. Repro

    python scripts/run_lowmem.py BDMAP_00000031 1   # stages 1..2d
    python scripts/run_lowmem.py BDMAP_00000031 2   # stage 2e
    python scripts/run_lowmem.py BDMAP_00000031 3   # 2f + 2g + write
    python postprocessing_vertebrae.py --case BDMAP_00000006 \
        --pred_dir AbdomenAtlasDemoPredict --ct_root data/AbdomenAtlasDemo \
        --out_dir out_v9 --report_dir reports/v9
    python scripts/diag_blades.py CT RAW.nii.gz RAW OUT.nii.gz v9 OUTDIR
    python scripts/test_2g_direct.py SEG CT OUT   # stage-2g A/B harness
    python tests/test_arch_phantom.py             # hier+core gated PASS

Pipeline QA: reports/v9/*_postprocessing_qa.json ("imbrication" block =
stage 2g record incl. gate values; "bands" records carry the per-band upv
before/after that convicted the hier ablation).
