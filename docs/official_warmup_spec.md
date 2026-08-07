# Official warm-up spec (SuPreM "Apply to Vertebrae Segmentation")

Source: the "here" link in Call4Research.pdf (https://www.cs.jhu.edu/~zongwei/advert/Call4Research.pdf).

## 0. Data
```bash
wget http://www.cs.jhu.edu/~zongwei/dataset/AbdomenAtlasDemo.tar.gz
tar -xzvf AbdomenAtlasDemo.tar.gz
```
Layout: `AbdomenAtlasDemo/BDMAP_00000006/ct.nii.gz`, `AbdomenAtlasDemo/BDMAP_00000031/ct.nii.gz` (CT only; masks come from inference).

## 1. Repo + checkpoint
```bash
git clone https://github.com/MrGiovanni/SuPreM
cd SuPreM/direct_inference/pretrained_checkpoints/
wget http://www.cs.jhu.edu/~zongwei/model/swin_unetr_totalsegmentator_vertebrae.pth
cd ..
```
Add `--no-check-certificate` on certificate errors.

## 2. Environment (official pins)
conda python=3.9; torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0; monai[all]==0.9.0; `pip install -r requirements.txt`.

## 3. Inference
```bash
datarootpath=/path/to/AbdomenAtlasDemo   # NEED MODIFICATION
pretrainpath=./pretrained_checkpoints/swin_unetr_totalsegmentator_vertebrae.pth
savepath=./AbdomenAtlasDemoPredict
python -W ignore inference.py --save_dir $savepath --checkpoint $pretrainpath \
    --data_root_path $datarootpath --customize
```
Output per case: `combined_labels.nii.gz` + `segmentations/vertebrae_{C1..C7,T1..T12,L1..L5}.nii.gz`.

## 4. Post-process (the graded part)
Inspect `combined_labels.nii.gz` over `ct.nii.gz` in ITK-SNAP. The predictions contain many errors
(fragments, holes, merged or mislabeled adjacent levels). Design automatic post-processing in a file
named exactly `postprocessing_vertebrae.py` reducing as many errors as possible.

Anatomy: C1-C7 (cervical), T1-T12 (thoracic), L1-L5 (lumbar); sacrum/coccyx are not labeled classes.

## Hints from their Related Work
ShapeKit (Liu 2025, the lab's shape-refinement toolkit); Meng 2023 (graph optimization +
anatomic consistency cycle for vertebra ID); Jaus 2024 (full-body CT atlas); Seng 2026
(TotalSegmentator on osteoporotic vertebral fractures).

## Submission (from Call4Research.pdf)
Email zzhou82@jh.edu: CV + compressed refined `AbdomenAtlasDemoPredict` + `postprocessing_vertebrae.py`.
Max one submission per week; rolling review.
