# JHU BodyMaps Research Assistant: Full Prep Guide

Prepared for Abhijit. Two parts. Part 1 makes you understand the research you are applying to join, built from the seminar "The AI That Sees Cancer Coming" (the video and slides the call PDF points to) plus the lab's papers and code. Part 2 is the exact warm-up assignment: what to submit, the pipeline to run, and a production-grade starter script. Read Part 1 to be able to talk about the work; do Part 2 to produce the deliverables.

The lab: Zongwei Zhou (Assistant Research Professor, JHU Computer Science + Oncology, Sidney Kimmel Comprehensive Cancer Center; CCVL / BodyMaps). The application is the Research Assistant track. It is unpaid (JHU students can earn research units), roughly 35 to 40 hours per week over 9 to 12 months.

---

# PART 1: THE RESEARCH YOU ARE JOINING

## 1.1 The mission and why it matters

The lab builds AI that detects cancer from CT scans earlier than radiologists can, and is assembling a 3D atlas of the human body (BodyMaps) to make that possible. The clinical stakes, using pancreatic cancer as the running example:

- If pancreatic cancer is caught late, the 5-year survival rate is about 7%. If caught early, it rises to about 44%. Early detection is the whole game.
- About 80 million CT scans are performed yearly in the US. Even when a patient is already scanned, early tumors are missed roughly 60% of the time.
- For tumors under 2 cm, radiologist sensitivity is only about 33 to 44%.

So the opportunity is not exotic new imaging; it is reading the scans that already exist, better.

## 1.2 The flagship result: pancreatic early detection

The problem is framed as 3D semantic segmentation: classify every voxel as healthy pancreas, tumor, or background. Headline numbers from the reader study (validated against surgical or pathological outcomes, not just against radiologists):

| Metric | Radiologists | Their AI |
|---|---|---|
| Sensitivity, early tumors ≤ 2 cm | 33 to 44% | 94% |
| Sensitivity, all-size tumors | 76 to 92% | 97% |
| Specificity, normal patients | 82 to 96% | 99% |

Three things make this more than a table:

1. A 30-reader study (3 pancreas specialists, 12 general radiologists, 15 residents) confirmed the human sensitivity really is in the 30 to 40% range; the AI adds about +30% sensitivity at the same specificity. AI plus human beats either alone.
2. On pre-diagnostic scans (taken 3 to 36 months before the clinical diagnosis, all originally missed by radiologists), the AI detects the cancer a median of about 13.6 months earlier. Radiologist performance on that cohort is essentially zero.
3. The AI is a segmentation model, so it localizes the tumor, which makes it a reviewable assistant rather than a black-box alarm.

This is the proof-of-concept that everything else scales up from.

## 1.3 The three core challenges, and the three research thrusts

The whole program is organized around three obstacles to doing the above for every cancer, not just the pancreas. These map one-to-one onto the program's three thrusts and onto the themes in the recruiting email.

### Challenge A: tumors are tiny → thrust "Innovating Algorithms"

A tumor can be about 0.0001% of a 3D CT volume (versus objects that fill 5 to 50% of a 2D natural image). Standard detectors and even large general models struggle; it is a needle-in-a-haystack, Where's-Waldo problem. The lab's signature contribution here is UNet++ (Z. Zhou et al., IEEE TMI 2019; roughly 18,000 citations; associated with the AMIA Doctoral Dissertation Award). The insight: in a standard U-Net the optimal network depth for a given object size is not known in advance. UNet++ fixes this with two designs, (1) densely connected multi-scale feature aggregation via redesigned skip connections, and (2) deep supervision, which lets you prune the network to any depth at inference. Net effect: better segmentation across a wide range of object sizes, especially small ones, and faster inference. It has been adopted well beyond medicine.

### Challenge B: tumors are rare → thrust "Synthesizing Datasets"

Early-stage tumor scans are 10 to 20x less common than late-stage ones, and most early cases sit unrecognized in hospital archives. You cannot train a data-hungry model on data you do not have, so the lab synthesizes realistic tumors. This is the part that most directly touches "3D generative models" and "world models" from the email. The pipeline is two stages:

1. Shape, via cellular automata (Lai et al., MICCAI 2024). A simple-rule system (think Game of Life / Wolfram rules) grows tumor and organ shape over time using three base rules, growth, invasion, and death, plus biology-motivated rules like mass effect (the tumor pushes the organ boundary) and duct dilation (a blocked pancreatic duct fills with fluid and becomes visible). Zhou explicitly frames this as a world model: current state, action, next state. It is literally "the AI that simulates cancer coming," and it is a time machine in both directions, growing a small tumor into a large one or shrinking a large one back to its early appearance.
2. Texture, via diffusion models conditioned on the simulated tumor/vessel/duct/organ shapes, rendering the shape into a realistic CT.

Why synthetic data pays off, with numbers:
- It raises small-tumor (< 2 cm) sensitivity by about 5% (89% → 94%); the smallest lesion detected is about 2 mm, near the CT resolution limit (Q. Hu et al., CVPR 2023; Q. Chen et al., CVPR 2024).
- It reduces position bias. About 65% of pancreatic tumors arise in the head; models then miss body/tail tumors. You can synthesize tumors specifically where the model is weak (TextoMorph, X. Li et al., IEEE TMI 2026).
- It enables domain transfer. Collecting cancer cases at a new hospital is hard; collecting normal scans is easy. Inject synthetic tumors into the new hospital's normal scans and the model adapts to that scanner and population without new annotation. (Repos: github.com/MrGiovanni/SyntheticTumors and github.com/MrGiovanni/TextoMorph; the synthetic-tumor work ranked first in the Medical Segmentation Decathlon.)

### Challenge C: annotation is slow and expensive → thrust "Scaling Annotations"

Per-voxel tumor labeling needs busy expert radiologists; the private Felix pancreatic dataset (a CS + Radiology collaboration at Hopkins) represents on the order of 25 human-years for about 5,000 CT scans. Two moves attack this:

1. Data scaling laws, run downward. Instead of scaling data up, they measured the minimum needed: roughly half the real data already reaches the plateau performance, and with synthetic data you can get there with far less, on the order of a few hundred annotated scans and about 80% less annotation burden.
2. Active learning / human-in-the-loop, with a twist they call RC analysis. Standard active learning selects the most informative unlabeled scans (high entropy or high diversity) for experts to label. The twist: deliberately tune the AI to be over-sensitive (near-100% sensitivity, many false positives). Creating an annotation from scratch takes a radiologist 4 to 5 minutes; deleting a wrong AI pseudo-label takes about one second. So the human's job shifts from drawing to pruning, which is roughly 80% faster than annotating from scratch.

The payoff compounds: they annotated and released about 36,000 CT scans (the AbdomenAtlas line of datasets), which built the lab's reputation and attracted collaborators worldwide. They now have access to about 2.5 million CT scans from about 445 hospitals, useful for both training and, crucially, external testing across scanners and populations.

## 1.4 Chapter II: the aha moment, radiology reports as supervision

The step-by-step "one cancer at a time" plan (NIH funded about $2.5M over four years, starting with colon cancer) was working but slow. The breakthrough: use the radiology reports that already exist in every hospital as training signal, so you are no longer gated on per-voxel annotation. This is the vision-language part of the program, and it is directly adjacent to your own VLM background.

Why reports are the right signal:
- Writing the report is the radiologist's paid, accountable daily job, and a report covers every organ present, not just one. So it naturally supports many-cancer learning. Asking for per-voxel masks for hundreds of cancer types would be refused; reports come for free.
- A pointed critique of naive CLIP-style VLMs: in social media, image and caption are often not really paired ("Hawaii morning is beautiful" next to a selfie), whereas medical reports are trustworthy and detailed. And global image-text contrastive alignment fails when the object of interest is 0.0001% of the volume; you cannot align a whole-image feature to a report that is about a few voxels.

Their method, R-Super (learning segmentation from radiology reports; MICCAI 2025 best-paper honor): one language model extracts structured facts from the report (tumor count, size, location) and those facts supervise a segmentation model through a loss that penalizes mismatches (if the pseudo-mask says 2 cm but the report says 5 cm, penalize). With a large pile of image-report pairs plus a small pile of image-mask pairs, they trained the first model to segment seven cancer types at roughly radiologist level. For contrast, general medical foundation models (for example Merlin) score near random (AUC about 0.5) on this detection task.

Reports keep paying off in three more places:
- Sharper active learning: knowing the reported tumor size lets you apply a local threshold so the AI's prediction matches the report, merging fragmented detections into a clean mask a human can just edit.
- Hard-negative mining for false positives: many false alarms are non-cancer diseases (for example pancreatitis, an inflamed pancreas that mimics early cancer). Reports let you automatically find these rare confusers and fine-tune them away, even when the confuser is rarer than the cancer itself.
- Grounded report generation (RadGPT, ICCV 2025, part of AbdomenAtlas 3.0): standard end-to-end image-to-report VLMs hallucinate fluent but wrong reports. Their two-stage approach first segments (tumor, vessels, organ, location), then writes the report grounded on that structured representation (a segmentation map, JSON, or bounding boxes an LLM can read). Segmentation and reporting reinforce each other in a loop. This matters because segmentation is not the radiologist's end goal; the report is.

## 1.5 From detection to a "cancer world model"

The framing that ties the whole talk together, and the phrase to remember, is the cancer world model. The clinical-deployment reality check is PPV (positive predictive value): if the model flags a tumor, how often is it right. In a general population where cancer prevalence is about 0.01%, even 95% sensitivity and 99% specificity give a PPV under 1% (thousands of false alarms per true case). The fix is to apply the model to high-risk populations where prevalence is higher (about 4%), pushing PPV up to roughly 80%, which beats FDA-cleared breast-screening software (about 20 to 40%). This drives three ongoing directions:

1. Forecast cancer before it is visible, by combining EHR risk factors with imaging biomarkers to define the high-risk population.
2. Treat, not just detect: feed the segmentation into radiotherapy planning (destroy the tumor, spare healthy tissue) and predict treatment response.
3. Make CT itself safer: extract more from less radiation via sparse-view reconstruction and image enhancement.

The unifying vision: a predictive model that takes a diseased patient as input and outputs a healthy patient, predicting how a cancer will develop, why it began, and how to treat it. That is the "world model" language in the recruiting email, and it is the through-line from the cellular-automata simulator (Challenge B) to the clinical goal.

## 1.6 The map: people, datasets, code

- Director and mentors: Zongwei Zhou (director), Alan Yuille (JHU), Pedro R. A. S. Bassi (JHU postdoc), plus mentors at Johns Hopkins Medicine (Heng Li, Kai Ding), UCSF (Yang Yang, Kang Wang), Harvard (Arkadiusz Sitek), Northwestern (Ulas Bagci), NVIDIA (Yucheng Tang), and Jagiellonian University (Szymon Płotka).
- Datasets (AbdomenAtlas line): 1.0 (NeurIPS 2023, about 5,195 volumes, about 9 classes), 1.1 (used by SuPreM, 9,262 volumes, 25 classes), 2.0 (ICCV 2025, tumor-focused), 3.0 / RadGPT (ICCV 2025, adds reports).
- Code you will touch or cite: SuPreM (github.com/MrGiovanni/SuPreM), AbdomenAtlas (github.com/MrGiovanni/AbdomenAtlas), SyntheticTumors, TextoMorph, RadGPT, R-Super.

---

# PART 2: THE WARM-UP ASSIGNMENT

## 2.1 Where the warm-up fits

The warm-up is the entry point to "Scaling Annotations." You run the lab's pretrained 3D segmentation model on demo CT scans to produce predictions (the `AbdomenAtlasDemoPredict` folder), then improve one weak part of the output by writing `postprocessing_vertebrae.py`. That is exactly the human-in-the-loop cleanup step from Section 1.3, in miniature: the model produces imperfect labels, and you make them clean and trustworthy. Doing it well shows you can operate the segmentation-and-annotation machinery the whole lab runs on.

## 2.2 Exactly what you submit

Reply to Zongwei's email stating your preferred position (Research Assistant), and send three items to zzhou82@jh.edu:

1. Resume / CV.
2. A refined `AbdomenAtlasDemoPredict` folder, compressed.
3. `postprocessing_vertebrae.py`, the script that cleans the vertebrae segmentation.

Rules from the call: no fixed deadline, rolling review at month-end, at most one submission per week. So optimize for correctness, not speed.

## 2.3 Reading and watching list, in priority order

You have now covered the intro video and slides (Part 1 is your notes). Remaining, in order:

Tier 1, required to run the task (about half a day):
1. The Call4Research PDF, especially the warm-up paragraph and the "how to apply" list.
   https://www.cs.jhu.edu/~zongwei/advert/Call4Research.pdf
2. SuPreM repo README and the direct-inference guide (the code you run).
   https://github.com/MrGiovanni/SuPreM and https://github.com/MrGiovanni/SuPreM/blob/main/direct_inference/README.md
3. AbdomenAtlas repo README (case layout on disk, label conventions).
   https://github.com/MrGiovanni/AbdomenAtlas

Tier 2, to ground what you built on (about half a day):
4. SuPreM paper, "How Well Do Supervised 3D Models Transfer to Medical Imaging Tasks?" (ICLR 2024 oral). https://www.cs.jhu.edu/~zongwei/publication/li2023suprem.pdf and https://arxiv.org/abs/2501.11253
5. AbdomenAtlas paper (Medical Image Analysis 2024). https://arxiv.org/abs/2407.16697
6. Research statement. https://www.zongweiz.com/research

Tier 3, to speak to the vision (optional, about 1 to 2 hours):
7. UNet++ (IEEE TMI 2019) for the architecture story.
8. R-Super (github.com/MrGiovanni/R-Super) and RadGPT (github.com/MrGiovanni/RadGPT) for the report / VLM direction.
9. SyntheticTumors (github.com/MrGiovanni/SyntheticTumors) and TextoMorph (github.com/MrGiovanni/TextoMorph) for the synthesis / world-model direction.

## 2.4 The pipeline, step by step

Reuses the lab's own modules plus standard, state-of-the-art libraries rather than reinventing anything.

Environment. One CUDA GPU. About 12 GB VRAM is workable for the U-Net backbone; Swin UNETR wants more. No local GPU is fine; the demo set is small (Colab, Kaggle, or a cloud instance).

```bash
git clone https://github.com/MrGiovanni/SuPreM
cd SuPreM
conda create -n suprem python=3.9 -y
conda activate suprem
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 \
  --extra-index-url https://download.pytorch.org/whl/cu113
pip install "monai[all]==0.9.0"
pip install -r requirements.txt
```

If your GPU is a newer (CUDA 12) card and the pinned wheels will not run, move up to a compatible torch/monai pair and record the versions you used; reproducibility notes are the kind of detail the lab values.

Checkpoints.

```bash
cd direct_inference/pretrained_checkpoints/
wget https://huggingface.co/MrGiovanni/SuPreM/resolve/main/supervised_suprem_unet_2100.pth
wget https://huggingface.co/MrGiovanni/SuPreM/resolve/main/supervised_suprem_swinunetr_2100.pth
cd ../../
```

Demo data. Lay the demo CT scans out in the case-folder structure the inference script expects:

```
/path/to/AbdomenAtlasDemo/
    ├── BDMAP_00000001/ct.nii.gz
    ├── BDMAP_00000002/ct.nii.gz
    └── ...
```

If you do not yet have the tutorial's exact demo link (Section 2.7), the public mini set is a valid stand-in to build and test against:

```bash
huggingface-cli download AbdomenAtlas/AbdomenAtlas1.0Mini \
  --repo-type dataset --local-dir ./AbdomenAtlasDemo --token YOUR_HF_TOKEN
```

Confirm the exact demo the call expects before your final submission, so the folder name and cases match what they grade against.

Inference, to produce the prediction folder.

```bash
cd direct_inference/
python -W ignore inference.py \
  --save_dir ./AbdomenAtlasDemoPredict \
  --checkpoint ./pretrained_checkpoints/supervised_suprem_unet_2100.pth \
  --data_root_path /path/to/AbdomenAtlasDemo \
  --backbone unet \
  --store_result \
  --suprem
```

This writes predicted segmentations into `AbdomenAtlasDemoPredict/`, typically one subfolder per case with per-structure `.nii.gz` masks and/or a combined label map. Open a few in ITK-SNAP or 3D Slicer overlaid on the CT and look at the vertebra labels specifically before you script anything.

## 2.5 Writing `postprocessing_vertebrae.py`

Vertebra predictions are a classic weak spot: adjacent levels get swapped or merged, small fragments float off the spine, and a single vertebra can split into two labels. Your script should fix these without touching correct voxels. Strategy, each step targeting a real failure mode:

1. Largest component per label: for each vertebra label, keep only the largest 3D connected component (removes off-spine speckle).
2. Small-fragment removal: drop components below a voxel-count threshold.
3. Level ordering: vertebrae are monotonic along the superior-inferior (z) axis. Compute each label's centroid and flag ordering violations; relabel only clear violations, since a wrong relabel is worse than a flag.
4. Optional light morphological closing to fill 1 to 2 voxel gaps without bleeding into neighbors.

Reuse standard fast libraries rather than hand-rolling connected components: `connected-components-3d` (cc3d) for labeling, `nibabel` for IO, `numpy` / `scipy.ndimage` for centroids and morphology. Run over the whole folder and preserve each file's affine and dtype.

```python
"""Post-process predicted vertebra segmentations.

Cleans common failure modes in per-vertebra CT segmentation: keeps the
largest connected component per label, drops small fragments, and flags
superior-inferior level-ordering violations. Reuses cc3d (fast 3D
connected components), nibabel (NIfTI IO), and scipy.ndimage.
"""

import argparse
import logging
from pathlib import Path

import cc3d
import nibabel as nib
import numpy as np
from scipy import ndimage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("postprocessing_vertebrae")

# Replace with the tutorial's actual vertebra label ids and voxel spacing.
VERTEBRA_LABELS = list(range(1, 25))
MIN_COMPONENT_VOXELS = 200
CLOSING_ITERATIONS = 1


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask
    labeled = cc3d.connected_components(mask.astype(np.uint8), connectivity=26)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    return labeled == counts.argmax()


def remove_small_fragments(mask: np.ndarray, min_voxels: int) -> np.ndarray:
    labeled = cc3d.connected_components(mask.astype(np.uint8), connectivity=26)
    keep = np.zeros_like(mask, dtype=bool)
    for cid, count in enumerate(np.bincount(labeled.ravel())):
        if cid == 0 or count < min_voxels:
            continue
        keep |= labeled == cid
    return keep


def clean_label_map(seg: np.ndarray) -> np.ndarray:
    out = np.zeros_like(seg)
    centroids = {}
    for label in VERTEBRA_LABELS:
        mask = seg == label
        if not mask.any():
            continue
        mask = keep_largest_component(mask)
        mask = remove_small_fragments(mask, MIN_COMPONENT_VOXELS)
        if CLOSING_ITERATIONS > 0:
            mask = ndimage.binary_closing(mask, iterations=CLOSING_ITERATIONS)
        if not mask.any():
            continue
        out[mask] = label
        centroids[label] = ndimage.center_of_mass(mask)
    _flag_ordering_violations(centroids)
    return out


def _flag_ordering_violations(centroids: dict) -> None:
    """Log labels whose z-centroid breaks the expected superior-inferior order.

    Report only; automatic relabeling is left conservative and opt-in,
    since a wrong relabel is worse than a flagged one.
    """
    ordered = sorted(centroids.items())
    z_values = [c[2] for _, c in ordered]  # assumes z is axis 2; verify on data
    for (label, _), z_prev, z_curr in zip(ordered[1:], z_values[:-1], z_values[1:]):
        if z_curr > z_prev:  # direction depends on orientation; verify on a real case
            logger.warning("Label %s breaks expected z-ordering", label)


def process_case(seg_path: Path, out_path: Path) -> None:
    img = nib.load(str(seg_path))
    seg = np.asarray(img.dataobj)
    cleaned = clean_label_map(seg)
    out_img = nib.Nifti1Image(cleaned.astype(seg.dtype), img.affine, img.header)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_img, str(out_path))
    logger.info("Wrote %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-process vertebra segmentations.")
    parser.add_argument("--pred_dir", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--seg_name", default="combined_labels.nii.gz",
                        help="Vertebra label-map filename within each case folder.")
    args = parser.parse_args()

    cases = sorted(p for p in args.pred_dir.iterdir() if p.is_dir())
    if not cases:
        logger.error("No case folders found under %s", args.pred_dir)
        return
    for case in cases:
        seg_path = case / args.seg_name
        if not seg_path.exists():
            logger.warning("Skipping %s: %s not found", case.name, args.seg_name)
            continue
        process_case(seg_path, args.out_dir / case.name / args.seg_name)


if __name__ == "__main__":
    main()
```

Before finalizing: match `VERTEBRA_LABELS`, the per-case filename, and the z-axis assumption to what the demo predictions actually contain (check one file's shape, affine, and unique label values). Verify the ordering direction on a real case. Keep the edit conservative; the grader will look at whether you removed genuine errors without damaging correct anatomy.

## 2.6 Pitfalls

- Torch/MONAI version drift on newer GPUs: move up a compatible pair rather than fighting the pins; record versions.
- Orientation and affine: always carry the original affine and header into outputs; a cleaned mask with a dropped affine is unusable.
- Label convention: confirm whether predictions are one combined multi-label file or per-structure binaries, and which ids are vertebrae, before scripting.
- Over-cleaning: aggressive closing or thresholds erase thin true structures; tune on one case with a viewer open.
- Whole-volume OOM: use the provided sliding-window inference path.
- Bad case: log and skip malformed cases rather than crashing the batch.

## 2.7 The one thing to confirm from the PDF

The Call4Research PDF's warm-up sentence, "please take a look at our warm-up training described here," has "here" as a hyperlink I could not resolve to an exact URL from a fetch. That link is the authoritative tutorial and will name the exact demo dataset and expected output format. To grab it: open the PDF in a browser, right-click "here" and copy the link. Follow it over this guide wherever they differ. If you paste that link (or attach the PDF), I will align every command, the demo download, and the label ids in the script to match it exactly.

---

# PART 3: HOW TO POSITION YOURSELF

Your background lines up unusually well with this lab. Concrete hooks for your reply email and any interview:

- World models: Zhou frames the cellular-automata tumor simulator as a world model (state, action, next state) and the program's end goal as a "cancer world model." Your world-models work (literature, pretraining and post-training paradigms, simulative and video-generation world models) is directly relevant to Synthesizing Datasets and to the forecast-and-treat direction. This is your strongest single hook.
- Vision-language models and VLM interpretability: maps straight onto Chapter II (R-Super's report-grounded supervision, RadGPT's grounded, hallucination-resistant reporting). Your interpretability angle is relevant to making report supervision trustworthy.
- Semi-supervised medical image analysis: maps onto the data-scaling-laws and active-learning story.
- Conformal / uncertainty methods: a clean niche you could pitch, since the deployment story is all about false positives, PPV, and choosing which scans a human should review (entropy and diversity selection). Uncertainty-aware selection and calibrated PPV are a natural contribution.

Reply tone: concise and technical. State the position, that you completed the warm-up, one line on what your post-processing improved (for example, "removed off-spine fragments and enforced level ordering, correcting N mislabeled vertebrae across the demo set"), and that details are in an attached note. Add one sentence connecting your world-models or VLM background to their direction; keep it short.

## Submission checklist

- [ ] CV / resume as PDF.
- [ ] `AbdomenAtlasDemoPredict/` produced by inference, then refined, zipped.
- [ ] `postprocessing_vertebrae.py` that runs end to end and preserves affines.
- [ ] A short note: what you ran, env versions, what the post-processing fixed, one before/after example.
- [ ] Reply to zzhou82@jh.edu stating Research Assistant and attaching the three items.

## Suggested schedule

- Day 1: read Tier 1, skim Tier 2; set up the environment; download checkpoints and demo data.
- Day 2: run inference, produce `AbdomenAtlasDemoPredict`, inspect vertebra failures in a viewer.
- Day 3: write and tune `postprocessing_vertebrae.py`, generate the refined folder, write the note, reply to the email.

The core task is small and very doable for you. The signal they want is a clean, reproducible run plus a sensible, conservative fix to a real weak spot, from someone who clearly understands the research it feeds into.
