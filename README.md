# Hybrid Radiomic Feature Fusion for Lung Disease Classification

Semester project for **AI Lab** 

Classifies **Normal / Bacterial Pneumonia / Viral Pneumonia** from chest X-ray images using **hybrid radiomic features** (statistical + GLCM texture + Gabor filters) fused together, reduced with PCA, and classified with:

- Logistic Regression
- Naïve Bayes (best performer ~63–64%)
- GPU-accelerated SVM (OvR)

## Key Results (80-20 split)

| Model              | Accuracy | Notes                             |
|--------------------|----------|-----------------------------------|
| Naïve Bayes        | 63.26%   | Fastest & most stable             |
| Logistic Regression| 53.78%   | Stable baseline                   |
| SVM                | 51.09%   | Overfits – needs more tuning      |

## Repository Structure

- `notebooks/`       → exploration & experiments  
- `data/`            → link on where the dataset is  
- `results/`         → figures, metrics, small models  
- `reports/`         → final PDF report not yet there if someone wants it message me  

## Dataset

Chest X-ray images (Normal / Bacterial / Viral Pneumonia)
→ Not included due to size → download from Kaggle / original source and place in data/file.txt


## How to Run (Quick Start)

```bash
# 1. Clone repo
git clone https://github.com/AAB-69/lung-disease-radiomics-ml.git
cd lung-disease-radiomics-ml

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Recommended) use conda
conda env create -f environment.yml
conda activate lung-radiomics

# 4. Run main script (example)
python src/train_evaluate.py --split 0.8

```

Made with ❤️ for educational purposes.
