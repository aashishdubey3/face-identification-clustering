# 👁️ Robust Face Identification & Clustering Pipeline

An automated, end-to-end computer vision system designed to ingest unorganized datasets of human faces, extract high-dimensional facial embeddings, and cluster identical individuals together regardless of varied lighting, extreme angles, or spontaneous expressions.

## ⚙️ Architecture & Strategy

To achieve robustness and handle edge cases, traditional cascades were bypassed in favor of a modern deep-learning pipeline:

* Face Detection: RetinaFace - Chosen for its state-of-the-art robustness. It reliably detects faces even in poor lighting conditions or severe profile angles.
* Feature Extraction: ArcFace - Utilized to generate high-dimensional numerical embeddings. ArcFace maximizes the margin between different identities in the vector space, making clustering significantly more accurate.
* Clustering Engine: DBSCAN (Density-Based Spatial Clustering of Applications with Noise) - Selected because the number of unique individuals (K) is completely unknown at runtime. It also effectively filters out false positives as noise.
* Confidence Scoring: Calculated using Cosine Similarity. The system finds the mathematical centroid of each cluster and measures the cosine distance of each individual image embedding to that center.

## 🚀 Installation & Setup

1. Clone the repository and navigate to the project directory.

2. Install the required dependencies:
   pip install deepface opencv-python scikit-learn numpy tf-keras

3. Prepare your data:
   Create a directory named `person_identification` in the root folder and place your raw, unorganized images (.jpg, .jpeg, .png) inside.

## 💻 Usage

Execute the main pipeline:
   python main.py

### 📂 Output Structure & "User-First" Design
The system automatically creates a `Clustered_Results` directory containing sub-folders for each identified individual. 

To prioritize user experience and transparency, the pipeline dynamically calculates the confidence score of each match and appends it directly to the output filename (e.g., `98.5%_Match_image_01.jpg`). This allows end-users to sort by name and immediately assess the model's certainty without needing to parse external log files.
