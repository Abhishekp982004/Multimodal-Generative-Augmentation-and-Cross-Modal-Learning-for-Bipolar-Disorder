# 🎧 Multimodal Speech Emotion Recognition (Audio + Text)

This project presents a **Multimodal Speech Emotion Recognition (SER)** system that combines **audio acoustic features** with **textual emotion descriptions** to improve emotion classification performance.  
By fusing complementary information from both modalities, the proposed approach significantly outperforms unimodal baselines.

---

## 📌 Project Motivation

Traditional speech emotion recognition systems rely only on acoustic features, which may fail to capture semantic or contextual cues.  
To overcome this limitation, this project integrates:

- **Audio-based emotional characteristics**
- **Text-based semantic emotion descriptions**

The fusion enables more robust and interpretable emotion recognition.

---

## 🎯 Objectives

- Extract meaningful **audio features** from speech signals  
- Perform **audio data augmentation** to improve generalization  
- Generate **emotion-aware textual descriptions**  
- Convert text into numerical embeddings using **TF–IDF**  
- Fuse audio and text representations  
- Train a **multimodal machine learning model**  
- Compare performance with unimodal baselines  

---

## 🗂️ Dataset

- Speech emotion dataset with labels:
  - ANG — Angry  
  - HAP — Happy  
  - SAD — Sad  
  - FEA — Fear  
  - NEU — Neutral  

- Audio format: `.wav`
- Labels extracted directly from filenames

---

## ⚙️ System Architecture

The pipeline processes raw audio files through a multi-stage workflow involving feature extraction, data augmentation, and NLP-based feature fusion for final classification.

- **Audio (.wav)**
    - **Feature Extraction**
        - `RMS` (Root Mean Square Energy)
        - `ZCR` (Zero Crossing Rate)
        - `Spectral Centroid`
        - `Spectral Bandwidth`
        - `Spectral Rolloff`
        - `MFCC` (Mel-frequency Cepstral Coefficients)
    - **Audio Augmentation**
        - Pitch Shift
        - Time Stretch
        - Noise Injection
    - **Text Description Generation**
        - Emotion-aware sentences
    - **Text Embedding**
        - TF–IDF Vectorization
    - **Feature Fusion**
        - Audio + Text Concatenation
    - **Classification**
        - Random Forest Classifier


---

## 🔊 Audio Feature Extraction

Each speech sample is represented using:

| Feature | Description |
|------|------|
| RMS | Signal energy / loudness |
| ZCR | Voice excitation and sharpness |
| Spectral Centroid | Brightness of sound |
| Spectral Bandwidth | Frequency spread |
| Spectral Rolloff | High-frequency dominance |
| MFCC | Perceptual timbre features |

---

## 🔁 Data Augmentation Techniques

To increase data diversity and reduce overfitting:

- **Pitch Shifting** – simulates different vocal tones  
- **Time Stretching** – simulates speaking speed variations  
- **Noise Injection** – improves robustness to real-world noise  

Augmentation increased dataset size to nearly **30,000 samples**.

---

## 📝 Text Description Generation

For each audio sample, a corresponding text description is generated using:

- Neutral base sentences  
- Emotion-specific modifiers  
- Controlled randomness to avoid label leakage  

Example:“The speaker says a short sentence with a slightly strong tense tone.”
This simulates semantic emotion cues without using real transcripts.

---

## 🔤 Text Embedding

Text descriptions are converted into numerical representations using:

- **TF–IDF Vectorizer**
- Limited feature dimensions for fast computation
- Efficient CPU-based embedding generation

---

## 🔗 Multimodal Feature Fusion

Final feature vector is constructed as:X = [Audio Features | Text Embeddings]
- Early fusion strategy
- StandardScaler applied before training

---

## 🤖 Model Training

- Classifier: **Random Forest**
- Balanced class weights
- Stratified train–test split
- Hyperparameters tuned for stability and accuracy

---

## 📊 Results

| Model | Accuracy |
|------|------|
| Audio Only | ~62% |
| Text Only | ~39% |
| **Multimodal (Audio + Text)** | **~83%** |

### ✅ Key Observation

The multimodal model significantly outperforms unimodal approaches, confirming that **audio and text provide complementary emotional cues**.

---

## 📈 Visualizations Included

- Waveform & spectrogram comparison (original vs augmented)
- Audio feature distributions
- Pairplots across emotions
- Correlation heatmap
- Feature importance ranking
- RMS distribution after augmentation
- Final accuracy comparison plot

---

## 🚀 Technologies Used

- Python
- Librosa
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## 🔮 Future Work

- Use pretrained audio models (Wav2Vec2, HuBERT)
- Replace TF–IDF with transformer-based text embeddings
- Integrate real speech transcripts
- Extend system for real-time emotion recognition
- Deploy as a web-based emotion analysis tool

---

## 👤 Authors

- Abhishek P
- Harsha
- Lohit J
  
CSE (AIML)  
PES University  

---



This project was developed as part of academic course work of Advanced Foundations for ML-UE23AM342AA1

