# 📋 PROJECT SPECIFICATION
## Dicoding Final Project — Sistem Machine Learning MLOps

- **Project Type:** Machine Learning Operations (MLOps)
- **Development Environment:** Antigravity IDE
- **Primary Language:** Python
- **Recommended Python:** 3.12.7
- **Recommended MLflow:** 2.19.0
- **Target:** Dicoding Final Project / Submission
- **GitHub Username:** `NaufalArkaan`
- **Arsitektur:** Experiment → Preprocessing → Training → MLflow → CI → Docker → Serving → Prometheus → Grafana → Alerting

---

## 1. ROLE

Anda bertindak sebagai **Senior MLOps Engineer** dan **Senior Software Engineer**.

Tugas Anda adalah membantu membangun project akhir Dicoding secara end-to-end, maintainable, reproducible, dan sesuai dengan seluruh rubric submission yang diberikan.

Jangan hanya membuat kode yang "bisa berjalan".

Project harus memenuhi:

- Struktur file yang diminta Dicoding.
- Workflow MLOps yang benar.
- Requirement setiap kriteria.
- Bukti/screenshot yang dibutuhkan.
- Reproducibility.
- Tidak ada credential yang hardcoded.
- Tidak ada error pada workflow yang diwajibkan.
- Repository Kriteria 1 dan Kriteria 3 dapat diperiksa reviewer.

---

## 2. ATURAN PALING PENTING

### 2.1 Minimum Passing Requirement

Terdapat 4 kriteria:

| Kriteria | Deskripsi |
|---|---|
| Kriteria 1 | Eksperimen Dataset |
| Kriteria 2 | Model Machine Learning |
| Kriteria 3 | Workflow CI |
| Kriteria 4 | Monitoring & Logging |

Setiap kriteria **WAJIB** mendapatkan minimal 2 points.

Jika:

```
K1 = 0  atau  K2 = 0  atau  K3 = 0  atau  K4 = 0
```

maka submission **REJECTED**.

Oleh karena itu, jangan mengorbankan requirement Basic hanya untuk mengejar Advanced.

---

## 3. STRATEGI IMPLEMENTASI

Implementasikan project secara bertahap:

```
PHASE 1   Dataset
PHASE 2   Experimentation + EDA
PHASE 3   Automated Preprocessing
PHASE 4   Model Training + MLflow
PHASE 5   Hyperparameter Tuning
PHASE 6   MLflow/DagsHub
PHASE 7   MLProject
PHASE 8   GitHub Actions CI
PHASE 9   Docker
PHASE 10  Model Serving
PHASE 11  Prometheus
PHASE 12  Grafana
PHASE 13  Alerting
PHASE 14  Submission Packaging
```

Jangan langsung membuat seluruh project sekaligus. Setiap phase harus diuji terlebih dahulu sebelum melanjutkan ke phase berikutnya.

---

## 4. DEVELOPMENT ENVIRONMENT

Gunakan environment berikut:

```
Python 3.12.7
mlflow==2.19.0
```

Dependency lain harus disesuaikan dengan dataset dan model yang akhirnya digunakan.

Gunakan virtual environment:

```bash
python -m venv .venv
```

Jangan mengandalkan global Python packages.

---

## 5. DATASET

Dataset yang digunakan pada project ini sudah ditentukan sebagai berikut:

| Placeholder | Nilai |
|---|---|
| `DATASET_NAME` | Bank Marketing Dataset |
| `DATASET_SOURCE` | UCI Machine Learning Repository (Bank Marketing Data Set) |
| `DATASET_FILE` | `bank-full.csv` |
| `TARGET_COLUMN` | `y` (apakah nasabah berlangganan deposito berjangka: yes/no) |
| `TASK_TYPE` | Klasifikasi Biner (Binary Classification) |

Sebelum implementasi modelling, identifikasi:

- nama dataset → **Bank Marketing (bank-full)**
- sumber dataset → UCI ML Repository
- format dataset → CSV, delimiter `;`
- jumlah data → ± 45.211 baris (bank-full.csv)
- fitur → age, job, marital, education, default, balance, housing, loan, contact, day, month, duration, campaign, pdays, previous, poutcome
- target → `y` (yes/no)
- tipe data → campuran numerik & kategorikal
- missing values → perlu dicek (dataset ini umumnya menandai unknown sebagai kategori tersendiri pada beberapa kolom, bukan NaN eksplisit)
- duplicate → perlu dicek ulang saat EDA
- outlier → perlu dicek terutama pada `balance`, `duration`, `campaign`
- categorical features → job, marital, education, default, housing, loan, contact, month, poutcome
- numerical features → age, balance, day, duration, campaign, pdays, previous
- distribusi target → **imbalanced** (kelas "no" jauh lebih dominan daripada "yes") — perlu penanganan khusus (misalnya class_weight, resampling, atau evaluasi dengan metrik selain accuracy)
- potensi data leakage → kolom `duration` sangat berkorelasi dengan target karena durasi panggilan hanya diketahui setelah panggilan selesai; perlu dipertimbangkan apakah kolom ini dihilangkan atau digunakan dengan catatan pada notebook

Raw dataset dan processed dataset harus dibedakan:

- Raw: `bank-full_raw.csv` (atau `bank-full.csv` asli, tidak diubah)
- Processed: `bank-full_preprocessing.csv` (hasil preprocessing)

---

## 6. KRITERIA 1 — EXPERIMENTATION

### Objective

Melakukan eksperimen manual menggunakan template eksperimen MSML sebelum membuat automation preprocessing.

Template yang menjadi dasar mempunyai struktur:

1. Perkenalan Dataset
2. Import Library
3. Memuat Dataset
4. Exploratory Data Analysis (EDA)
5. Data Preprocessing

Template juga menyatakan bahwa dataset dapat berasal dari public repository seperti Kaggle, UCI ML Repository, Open Data, atau data primer. **Dataset yang dipilih: UCI ML Repository — Bank Marketing (bank-full).**

---

## 7. NOTEBOOK EXPERIMENT

File:

```
preprocessing/Eksperimen_NaufalArkaan.ipynb
```

Notebook WAJIB mengikuti struktur dasar template.

### Section 1 — Perkenalan Dataset

Jelaskan:

- nama dataset → Bank Marketing Dataset (bank-full)
- sumber → UCI Machine Learning Repository
- tujuan dataset → memprediksi apakah nasabah bank akan berlangganan deposito berjangka
- problem statement → bagaimana memprediksi kecenderungan nasabah untuk subscribe term deposit berdasarkan data kampanye pemasaran telepon
- machine learning task → klasifikasi biner
- target variable → `y`
- jumlah data → ± 45.211 baris
- fitur → 16 fitur (7 numerik, 9 kategorikal)

### Section 2 — Import Library

Import hanya library yang benar-benar digunakan.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

Tambahkan library lain sesuai kebutuhan (misalnya `sklearn.preprocessing`, `sklearn.model_selection`).

### Section 3 — Memuat Dataset

Notebook harus melakukan data loading secara eksplisit dari `bank-full.csv` (perhatikan delimiter `;` pada dataset asli UCI).

Lakukan validasi:

```python
df = pd.read_csv("bank-full.csv", sep=";")
df.head()
df.shape
df.info()
df.describe()
```

---

## 8. EDA

EDA harus benar-benar dilakukan, bukan hanya formalitas.

Minimal analisis untuk dataset **bank-full**:

- Dataset shape
- Column information (tipe data setiap kolom)
- Missing values (termasuk pengecekan nilai `"unknown"` pada kolom kategorikal)
- Duplicate values
- Statistical summary (`describe()`)
- Target distribution (`y`) — cek tingkat imbalance
- Feature distribution (age, balance, duration, campaign, dll.)
- Korelasi antar fitur numerik, serta hubungan fitur kategorikal terhadap target (misalnya `job` vs `y`, `poutcome` vs `y`)

Visualisasi harus disesuaikan dengan dataset. Jangan membuat visualisasi yang tidak memiliki tujuan analitis. Setiap visualisasi harus memiliki interpretasi.

Contoh:

```
Insight:
- Distribusi target sangat tidak seimbang, mayoritas nasabah tidak subscribe (kelas "no").
- Kolom `duration` memiliki korelasi tinggi terhadap target namun berpotensi menyebabkan data leakage.
- Nasabah dengan `poutcome = success` pada kampanye sebelumnya cenderung lebih mudah subscribe.
```

---

## 9. DATA PREPROCESSING

Preprocessing harus berdasarkan karakteristik dataset **bank-full**. Jangan melakukan preprocessing secara asal.

Kemungkinan tahap yang relevan untuk dataset ini:

- Penanganan nilai `"unknown"` pada kolom kategorikal (job, education, contact, poutcome)
- Duplicate Removal
- Outlier Handling (khususnya `balance`, `duration`, `campaign`)
- Encoding (One-Hot Encoding / Label Encoding untuk fitur kategorikal)
- Normalization / Standardization untuk fitur numerik
- Feature Selection (mempertimbangkan apakah `duration` tetap digunakan atau dihilangkan untuk menghindari leakage)
- Penanganan target imbalance (misalnya menggunakan `class_weight`, atau teknik resampling seperti SMOTE — jika digunakan, dilakukan **hanya pada data training**)

Template Dicoding menyebutkan tahapan tersebut sebagai contoh dan tidak terbatas pada itu. Gunakan hanya tahap yang relevan.

---

## 10. DATA LEAKAGE

Perhatikan data leakage.

Jika menggunakan Scaling, Encoding, Imputation, atau Feature selection yang membutuhkan fitting terhadap data, proses tersebut harus dilakukan dengan benar agar informasi validation/test tidak bocor ke training.

Jika melakukan train/test split:

```
Raw (bank-full.csv)
 ↓
Split (train/test)
 ↓
Fit preprocessing menggunakan training
 ↓
Transform training
 ↓
Transform testing
```

**Catatan khusus dataset ini:** Perhatikan kolom `duration` — sertakan pembahasan eksplisit di notebook mengenai risiko leakage kolom ini.

---

## 11. OUTPUT KRITERIA 1

Setelah preprocessing manual selesai, harus dihasilkan:

```
bank-full_preprocessing
```

Kemudian konversikan workflow manual menjadi:

```
automate_NaufalArkaan.py
```

jika mengejar Skilled/Advanced.

---

## 12. AUTOMATE PREPROCESSING

`automate_NaufalArkaan.py` harus:

- membaca raw dataset (`bank-full.csv`)
- menjalankan preprocessing
- menghasilkan processed dataset (`bank-full_preprocessing.csv`)
- menggunakan workflow preprocessing yang sama dengan notebook
- dapat dijalankan tanpa notebook
- dapat dijalankan ulang
- tidak bergantung pada state notebook

Contoh arsitektur:

```python
def load_data():
    ...

def clean_data():
    ...

def preprocess_data():
    ...

def save_processed_data():
    ...

def main():
    ...

if __name__ == "__main__":
    main()
```

---

## 13. KRITERIA 1 ADVANCED

Jika mengejar Advanced, buat GitHub Actions untuk preprocessing.

Workflow:

```
Trigger
 ↓
Checkout repository
 ↓
Setup Python
 ↓
Install dependencies
 ↓
Run preprocessing (bank-full.csv → bank-full_preprocessing.csv)
 ↓
Generate latest processed dataset
```

Workflow harus berhasil minimal satu kali tanpa error.

---

## 14. REPOSITORY KRITERIA 1

Nama repository:

```
Eksperimen_SML_NaufalArkaan
```

Link:

```
https://github.com/NaufalArkaan/Eksperimen_SML_NaufalArkaan
```

Visibility: **PUBLIC**

Struktur:

```
Eksperimen_SML_NaufalArkaan/
│
├── .workflow/
│
├── bank-full_raw
│
└── preprocessing/
    ├── Eksperimen_NaufalArkaan.ipynb
    ├── automate_NaufalArkaan.py
    └── bank-full_preprocessing
```

---

## 15. KRITERIA 2 — MODEL DEVELOPMENT

Model **WAJIB** menggunakan dataset hasil preprocessing (`bank-full_preprocessing`). Jangan menggunakan raw dataset untuk modelling.

Struktur:

```
Membangun_model/
├── modelling.py
├── modelling_tuning.py
├── bank-full_preprocessing
├── screenshoot_dashboard.jpg
├── screenshoot_artifak.jpg
├── requirements.txt
└── DagsHub.txt
```

---

## 16. MODELLING.PY

`modelling.py` adalah baseline training.

Untuk Basic:

```
Scikit-Learn + MLflow Tracking + MLflow autolog
```

MLflow harus berjalan secara lokal (`localhost` atau `127.0.0.1`).

Gunakan:

```python
mlflow.autolog()
```

Model dasar yang disarankan untuk klasifikasi biner pada dataset ini: `LogisticRegression`, `RandomForestClassifier`, atau `XGBoostClassifier` (pilih sesuai eksplorasi pada notebook).

---

## 17. MLflow TRACKING

Training harus menghasilkan informasi yang dapat dilihat di MLflow Tracking UI.

Minimal harus ada:

- Parameters
- Metrics (misalnya accuracy, precision, recall, f1-score — penting karena target imbalanced)
- Model
- Artifacts

Pastikan screenshot menunjukkan hasil yang valid.

---

## 18. HYPERPARAMETER TUNING

Jika mengejar Skilled/Advanced, buat:

```
modelling_tuning.py
```

File ini harus:

- melakukan hyperparameter tuning (misalnya `GridSearchCV` atau `RandomizedSearchCV`)
- melakukan manual MLflow logging
- mencatat metrik
- mencatat parameter
- melakukan model logging
- menghasilkan model/artifact

Manual logging diterapkan pada `modelling_tuning.py`, bukan mengganti requirement baseline `modelling.py`.

---

## 19. ADVANCED MLflow + DAGSHUB

Jika mengejar Advanced:

```
Training → MLflow → DagsHub → Online Tracking
```

Gunakan DagsHub sebagai remote MLflow tracking. Credential tidak boleh hardcoded — gunakan environment variable/secret.

Contoh konsep:

```python
import os
import dagshub
import mlflow

dagshub.init(
    repo_owner=os.environ["DAGSHUB_USERNAME"],
    repo_name=os.environ["DAGSHUB_REPO"],
    mlflow=True
)
```

> Placeholder yang perlu diisi user: `DAGSHUB_USERNAME`, `DAGSHUB_REPO`, `DAGSHUB_TOKEN` (belum ditentukan — tanyakan ke user sebelum implementasi tahap ini).

---

## 20. ADVANCED ARTIFACT

Untuk Advanced, tambahkan minimal 2 artefak tambahan selain artifact yang sudah dicakup pada tahap Skilled.

Contoh artefak tambahan yang relevan untuk dataset bank-full (klasifikasi biner imbalanced):

- `confusion_matrix.png`
- `feature_importance.png`
- `classification_report.json`
- `model_metadata.json`
- `prediction_sample.csv`

Artefak harus benar-benar dilog ke MLflow sebagai artifact.

---

## 21. KRITERIA 3 — MLPROJECT

Repository:

```
Workflow-CI
```

Link:

```
https://github.com/NaufalArkaan/Workflow-CI
```

Visibility: **PUBLIC**

Struktur:

```
Workflow-CI/
│
├── .workflow/
│
└── MLProject/
    ├── modelling.py
    ├── conda.yaml
    ├── MLProject
    ├── bank-full_preprocessing
    ├── Docker Hub link
    └── additional files
```

---

## 22. MLPROJECT FILE

`MLProject` harus mendefinisikan project MLflow. Harus mampu menjalankan training secara reproducible.

Konsep:

```
MLProject → conda.yaml → modelling.py → training → model artifact
```

---

## 23. GITHUB ACTIONS CI

Workflow harus dapat melakukan retraining ketika trigger dipicu.

Konsep:

```
GitHub Trigger
      ↓
Checkout
      ↓
Setup Python
      ↓
Install MLflow
      ↓
Run MLflow Project
      ↓
Training
      ↓
Generate Model
```

Pastikan workflow dibuat dengan benar dan diuji pada repository `NaufalArkaan/Workflow-CI`.

---

## 24. SECRETS

Jangan pernah hardcode:

- password
- token
- API key
- DagsHub token
- Docker Hub password
- GitHub token

Gunakan GitHub Secrets / Environment Variables.

Contoh:

```yaml
env:
  DAGSHUB_TOKEN: ${{ secrets.DAGSHUB_TOKEN }}
```

Jangan menuliskan credential asli di source code.

---

## 25. CI ARTIFACT STORAGE

Jika Skilled/Advanced, artifact hasil training harus disimpan pada repository/storage yang sesuai. Pilihan yang diperbolehkan:

- GitHub
- Google Drive
- GitHub LFS

Sesuaikan implementasi dengan workflow.

---

## 26. DOCKER — ADVANCED CI

Jika mengejar Advanced, workflow harus dapat menghasilkan Docker Image menggunakan `mlflow build-docker`.

Konsep:

```
GitHub Actions
      ↓
MLflow Project
      ↓
Train Model
      ↓
mlflow build-docker
      ↓
Docker Image
      ↓
Docker Hub
```

> Placeholder yang perlu diisi user: `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN` (belum ditentukan).

Image kemudian dapat digunakan untuk model serving.

---

## 27. KRITERIA 4 — MODEL SERVING

Model harus dapat di-serving pada local environment. Pilihan implementasi:

- MLflow Model Serving
- atau API/framework lain yang sesuai (misalnya Flask/FastAPI)

Harus ada bukti:

```
1.bukti_serving
```

Bukti harus menunjukkan model benar-benar berjalan (menerima input fitur nasabah bank dan mengembalikan prediksi `y`).

---

## 28. INFERENCE.PY

Buat:

```
Monitoring dan Logging/7.Inference.py
```

Fungsi utama:

```
Input (data nasabah) → Model → Prediction (yes/no) → Output
```

Inference harus dapat berkomunikasi dengan model serving.

---

## 29. PROMETHEUS

Buat:

```
2.prometheus.yml
3.prometheus_exporter.py
```

Exporter bertugas menyediakan metric yang dapat dibaca Prometheus.

Konsep:

```
Model/API → Exporter → Prometheus
```

---

## 30. METRICS

| Level | Jumlah Metric |
|---|---|
| Basic | ≥ 3 metrics berbeda |
| Skilled | ≥ 5 metrics berbeda |
| Advanced | ≥ 10 metrics berbeda |

Metrics harus relevan dengan sistem machine learning/model serving. Contoh kategori metric:

- Request count
- Prediction count
- Request latency
- Prediction latency
- Error count
- HTTP status count
- Request rate
- Prediction class distribution (proporsi prediksi "yes" vs "no")
- Model inference count
- Exception count

Jangan hanya membuat 10 nama metric yang sebenarnya mengukur hal yang sama tanpa makna.

---

## 31. GRAFANA

Grafana digunakan sebagai: **Visualization + Monitoring + Alerting**, mengambil metrics dari Prometheus.

Arsitektur:

```
Application → Exporter → Prometheus → Grafana
```

---

## 32. NAMA DASHBOARD

Ini **WAJIB**.

Nama dashboard Grafana harus menggunakan:

```
NaufalArkaan
```

Tujuannya agar screenshot dapat diverifikasi oleh reviewer. Jangan menggunakan nama generik seperti "Dashboard", "ML Monitoring", atau "Test Dashboard".

---

## 33. GRAFANA MONITORING

| Level | Target Metric |
|---|---|
| Basic | ≥ 3 metrics |
| Skilled | ≥ 5 metrics |
| Advanced | ≥ 10 metrics |

Metrik Grafana harus sesuai dengan metrics yang tersedia pada Prometheus.

---

## 34. GRAFANA ALERTING

| Level | Ketentuan |
|---|---|
| Basic | Alerting tidak menjadi syarat tambahan |
| Skilled | Minimal 1 alert |
| Advanced | Minimal 3 alerts |

Setiap alert harus mempunyai **Rule** + **Notification**. Bukti screenshot harus disimpan.

---

## 35. FINAL SUBMISSION STRUCTURE

Buat package final:

```
SMSML_NaufalArkaan.zip
```

Isi:

```
SMSML_NaufalArkaan/
│
├── Eksperimen_SML_NaufalArkaan.txt
│
├── Membangun_model/
│   ├── modelling.py
│   ├── modelling_tuning.py
│   ├── bank-full_preprocessing
│   ├── screenshoot_dashboard.jpg
│   ├── screenshoot_artifak.jpg
│   ├── requirements.txt
│   └── DagsHub.txt
│
├── Workflow-CI.txt
│
└── Monitoring dan Logging/
    ├── 1.bukti_serving
    ├── 2.prometheus.yml
    ├── 3.prometheus_exporter.py
    │
    ├── 4.bukti monitoring Prometheus/
    │   ├── 1.monitoring_<metriks>
    │   ├── 2.monitoring_<metriks>
    │   └── ...
    │
    ├── 5.bukti monitoring Grafana/
    │   ├── 1.monitoring_<metriks>
    │   ├── 2.monitoring_<metriks>
    │   └── ...
    │
    ├── 6.bukti alerting Grafana/
    │   ├── 1.rules_<metriks>
    │   ├── 2.notifikasi_<metriks>
    │   ├── 3.rules_<metriks>
    │   ├── 4.notifikasi_<metriks>
    │   └── ...
    │
    ├── 7.Inference.py
    └── additional files
```

Jangan membuat ZIP di dalam ZIP.

---

## 36. TXT FILE

### `Eksperimen_SML_NaufalArkaan.txt`

Berisi URL repository:

```
https://github.com/NaufalArkaan/Eksperimen_SML_NaufalArkaan
```

Repository harus **PUBLIC**.

### `Workflow-CI.txt`

Berisi URL:

```
https://github.com/NaufalArkaan/Workflow-CI
```

Repository harus **PUBLIC**.

---

## 37. REJECT CONDITIONS

Sebelum submission, lakukan audit.

**Kriteria 1**
- [ ] Template digunakan
- [ ] Experimentation manual
- [ ] Data loading
- [ ] EDA
- [ ] Preprocessing
- [ ] Notebook berhasil tanpa error

**Kriteria 2**
- [ ] ML model dibuat
- [ ] MLflow digunakan
- [ ] Tracking UI tersedia
- [ ] Artifact tersedia
- [ ] Logging tersedia

**Kriteria 3**
- [ ] MLProject tersedia
- [ ] GitHub Actions tersedia
- [ ] CI dapat menjalankan training

**Kriteria 4**
- [ ] Local serving berhasil
- [ ] Dashboard menggunakan username Dicoding (`NaufalArkaan`)
- [ ] Prometheus digunakan
- [ ] Grafana digunakan

Jika salah satu requirement dasar tersebut gagal, **STOP** dan perbaiki terlebih dahulu.

---

## 38. QUALITY GATE

Sebelum menyatakan project selesai, jalankan pemeriksaan:

| Gate | Deskripsi | Kondisi Lulus |
|---|---|---|
| Gate 1 | Notebook | Tidak ada error → PASS |
| Gate 2 | Preprocessing | Dataset berhasil dibuat → PASS |
| Gate 3 | MLflow | Run + Metrics + Model + Artifacts → PASS |
| Gate 4 | MLProject | `mlflow run` berhasil → PASS |
| Gate 5 | GitHub Actions | CI berhasil → PASS |
| Gate 6 | Docker | Image berhasil dibuat → PASS |
| Gate 7 | Serving | Inference berhasil → PASS |
| Gate 8 | Prometheus | Metrics tersedia → PASS |
| Gate 9 | Grafana | Dashboard + metrics → PASS |
| Gate 10 | Alert | Rule + notification → PASS |
| Gate 11 | Submission | ZIP structure valid → PASS |

---

## 39. DEVELOPMENT RULES UNTUK ANTIGRAVITY

**Rule 1 — Jangan mengarang informasi**

Informasi yang sudah tersedia:

| Item | Nilai |
|---|---|
| DATASET | Bank Marketing (bank-full.csv) |
| GITHUB USERNAME | NaufalArkaan |
| TARGET COLUMN | y |

Informasi yang **masih diperlukan** (jangan mengarang, tanyakan ke user):

- DAGSHUB USERNAME (jika mengejar Advanced)
- DOCKER HUB USERNAME (jika mengejar Advanced)
- MODEL final yang dipilih (ditentukan setelah eksperimen)

**Rule 2 — Jangan menghapus requirement**

Jangan menyederhanakan project dengan menghapus MLflow, MLProject, GitHub Actions, Prometheus, Grafana, Serving, Artifact, Screenshot, atau repository link hanya karena implementasinya lebih mudah.

**Rule 3 — Jangan hardcode credential**

Credential harus menggunakan environment variables/secrets.

**Rule 4 — Reproducibility**

Semua dependency harus dicatat menggunakan `requirements.txt`, dan untuk MLflow Project menggunakan `conda.yaml`.

**Rule 5 — Modular code**

Hindari satu file Python yang terlalu besar. Gunakan fungsi: `load_data()`, `preprocess_data()`, `train_model()`, `evaluate_model()`, `log_model()`, `save_artifacts()` sesuai kebutuhan.

---

## 40. IMPORTANT: DEVELOPMENT ORDER

Antigravity **HARUS** mengerjakan project secara bertahap. Jangan membuat semua file kosong sekaligus.

Urutan:

1. Inspect dataset (bank-full.csv)
2. Determine preprocessing
3. Complete experiment notebook
4. Run notebook
5. Validate notebook
6. Create automation preprocessing
7. Run preprocessing
8. Validate processed dataset
9. Create modelling.py
10. Run MLflow
11. Validate MLflow
12. Create tuning
13. Validate tuning
14. Configure DagsHub if Advanced
15. Create MLProject
16. Run MLflow Project locally
17. Create GitHub Actions
18. Test CI
19. Create Docker image
20. Test serving
21. Create Prometheus exporter
22. Test Prometheus
23. Configure Grafana
24. Create dashboard
25. Configure alerts
26. Capture evidence
27. Audit submission structure
28. Create final ZIP

---

## 41. DEFINITION OF DONE

Project **BELUM** dianggap selesai hanya karena aplikasi/model dapat berjalan.

Project baru dianggap selesai jika:

- [ ] Kriteria 1 minimum 2 pts
- [ ] Kriteria 2 minimum 2 pts
- [ ] Kriteria 3 minimum 2 pts
- [ ] Kriteria 4 minimum 2 pts
- [ ] Tidak ada kriteria bernilai 0
- [ ] Notebook berhasil
- [ ] MLflow berhasil
- [ ] MLProject berhasil
- [ ] GitHub Actions berhasil
- [ ] Serving berhasil
- [ ] Prometheus berhasil
- [ ] Grafana berhasil
- [ ] Bukti screenshot lengkap
- [ ] Dashboard menggunakan username Dicoding (`NaufalArkaan`)
- [ ] Repository Kriteria 1 PUBLIC (`Eksperimen_SML_NaufalArkaan`)
- [ ] Repository Kriteria 3 PUBLIC (`Workflow-CI`)
- [ ] Final ZIP sesuai struktur
- [ ] Tidak ada ZIP dalam ZIP
- [ ] Tidak ada credential yang bocor

---

## 42. TARGET LEVEL

Jika implementasi memungkinkan, gunakan pendekatan progressive enhancement:

```
┌───────────┐
│   BASIC   │  2 pts
└─────┬─────┘
      ↓
┌───────────┐
│  SKILLED  │  3 pts
└─────┬─────┘
      ↓
┌───────────┐
│ ADVANCED  │  4 pts
└───────────┘
```

Prioritas pertama adalah memastikan seluruh kriteria minimal Basic dan tidak ada reject condition. Setelah Basic aman, baru tingkatkan ke Skilled/Advanced.

---

## 43. INSTRUCTION UNTUK ANTIGRAVITY

Jangan langsung menulis seluruh kode project.

Pertama:

1. Analisis project requirement.
2. Analisis dataset bank-full (Bank Marketing).
3. Identifikasi task ML (klasifikasi biner).
4. Tentukan preprocessing yang sesuai.
5. Buat project plan.
6. Tampilkan struktur directory yang akan dibuat.
7. Identifikasi informasi yang masih diperlukan (DagsHub, Docker Hub, model final).
8. Setelah informasi lengkap, implementasikan phase pertama.
9. Jalankan/test.
10. Perbaiki error.
11. Baru lanjut ke phase berikutnya.

Setiap phase harus memiliki:

```
Implementation → Testing → Validation → Requirement Check → Next Phase
```

Jangan melompati testing.

---

## 44. FINAL PRINCIPLE

Project ini bukan hanya project Machine Learning. Project harus dipandang sebagai end-to-end MLOps pipeline:

```
Dataset (bank-full)
     ↓
Experiment + EDA
     ↓
Preprocessing
     ↓
ML Training + MLflow
     ↓
MLProject
     ↓
GitHub Actions
     ↓
Docker
     ↓
Model Serving
     ↓
Prometheus
     ↓
Grafana
     ↓
Alerting
```

Semua komponen harus saling terhubung dan dapat dibuktikan.
