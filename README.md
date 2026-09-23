# Klasifikasi Sampah — (AI Innovation)

Model klasifikasi gambar sampah organik vs anorganik, dikembangkan untuk
kompetisi KOMPRES 16 Informatika — Universitas Gunadarma, kategori AI Innovation.

**Demo live**: https://huggingface.co/spaces/rerefina/waste-classification

## Ringkasan Metode

Pipeline menggunakan pendekatan *transfer learning*: fitur visual gambar
diekstrak menggunakan CLIP ViT-L/14@336 (frozen, tidak dilatih ulang), lalu
sebuah classifier head (Logistic Regression) dilatih di atas embedding
tersebut untuk membedakan kelas Organik dan Anorganik.

**Dataset**: [Waste Classification Data](https://www.kaggle.com/techsash/waste-classification-data)
(Kaggle, oleh techsash) — ±25.000 gambar, 2 kelas (Organik/O, Anorganik/R).

**Hasil evaluasi**:
- Macro F1 pada test set: 0.9806
- 5-fold cross-validation pada train set: 0.9754 ± 0.0014

## Struktur Repo

- `extract_embeddings.py` — ekstraksi embedding CLIP dari dataset gambar
- `train_classifier.py` — training & evaluasi classifier (Logistic Regression)
- `app.py` — aplikasi demo Gradio (dideploy ke Hugging Face Spaces)
- `requirements.txt` — dependencies

## Cara Menjalankan

```bash
pip install -r requirements.txt

# 1. Ekstrak embedding (dataset harus punya struktur DATASET/TRAIN/<kelas>/ dan DATASET/TEST/<kelas>/)
python extract_embeddings.py --data_dir DATASET --out_dir embeddings

# 2. Train classifier
python train_classifier.py --emb_dir embeddings --out_dir model_out

# 3. Jalankan demo lokal
python app.py
```

## Teknologi

- Python, PyTorch, Transformers (Hugging Face)
- CLIP ViT-L/14@336 (`openai/clip-vit-large-patch14-336`)
- scikit-learn (Logistic Regression)
- Gradio (antarmuka web)
- Hugging Face Spaces / ZeroGPU (deployment)
- Google Colab (environment training)
