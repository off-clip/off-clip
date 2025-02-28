# OFF-CLIP: Improving Normal Detection Confidence in Radiology CLIP with Simple Off-Diagonal Term Auto-Adjustment

## About
OFF-CLIP (**OFF**-Diagonal **C**ontrastive **L**anguage-**I**mage **P**re-Training) is a pioneering refinement for radiology zero-shot classification, designed to boost normal case detection with minimal additional labeling. It leverages an innovative off-diagonal term loss to promote cohesive normal sample clustering while employing sentence-level text filtering to eliminate misaligned normal statements, thus reducing false negatives and positives.

![alt text](offclip_figure.png)

## Datasets

## How to start
```bash
pip install -r requirments.txt
```
## Train

## Validation
### Weight checkpoints  
Download the [offclip checkpoint](https://drive.google.com/file/d/1JmfB2jbl-58aBrxRwaMrGjhPNUUjKNC-/view?usp=drive_link) to test validation.

### Zero-shot classification for multi-label datasets
```bash
python3 validation.py --weight_path {weight path to load} --save_name {name to save similarities and results} -c configs/offclip.yaml
```

### Pointing game




