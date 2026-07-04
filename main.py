import os
import shutil
import numpy as np
from deepface import DeepFace
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURATION ---
DATASET_FOLDER = "person_identification"
OUTPUT_FOLDER = "Clustered_Results"

def clean_and_gather_images(folder_path):
    print(f"Scanning folder: '{folder_path}'...")
    allowed_extensions = ('.jpg', '.jpeg', '.png')
    valid_images = []

    if not os.path.exists(folder_path):
        print(f"Error: The folder '{folder_path}' doesn't exist!")
        return []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(allowed_extensions):
            full_path = os.path.join(folder_path, filename)
            valid_images.append(full_path)
    
    return valid_images

def extract_face_features(image_paths):
    print("\n======================================")
    print("      STARTING FACE SCANNING AI       ")
    print("======================================\n")
    
    all_embeddings = []
    successful_paths = []
    
    for path in image_paths:
        filename = os.path.basename(path)
        try:
            results = DeepFace.represent(
                img_path=path,
                model_name="ArcFace",
                detector_backend="retinaface",
                enforce_detection=True
            )
            face_embedding = results[0]["embedding"]
            all_embeddings.append(face_embedding)
            successful_paths.append(path)
            print(f"[{filename}] -> SUCCESS")
        except Exception:
            print(f"[{filename}] -> SKIPPED (No clear face found)")
            
    return np.array(all_embeddings), successful_paths

def group_and_score_faces(embeddings, image_paths):
    print("\n======================================")
    print("    GROUPING FACES & SCORING MATCHES  ")
    print("======================================\n")
    
    # 1. Group the faces. 'cosine' is the math used to compare fingerprints.
    # eps=0.4 is our strictness level. (Lower = stricter).
    clustering = DBSCAN(eps=0.4, min_samples=1, metric="cosine").fit(embeddings)
    labels = clustering.labels_
    
    # Create the output folder (delete it first if it already exists so we start fresh)
    if os.path.exists(OUTPUT_FOLDER):
        shutil.rmtree(OUTPUT_FOLDER)
    os.makedirs(OUTPUT_FOLDER)
    
    # Organize data by person
    people_groups = {}
    for i, label in enumerate(labels):
        if label not in people_groups:
            people_groups[label] = []
        people_groups[label].append({
            "path": image_paths[i],
            "embedding": embeddings[i]
        })

    # 2. Calculate Confidence Scores and Save
    person_counter = 1
    for label, items in people_groups.items():
        # If the AI couldn't figure out who this is, label is -1
        if label == -1:
            person_name = "Uncategorized_Noise"
        else:
            person_name = f"Person_{person_counter}"
            person_counter += 1
            
        person_dir = os.path.join(OUTPUT_FOLDER, person_name)
        os.makedirs(person_dir)
        print(f"Creating folder for {person_name}...")
        
        # Find the mathematical "center" of this person's face group
        group_embeddings = [item["embedding"] for item in items]
        centroid = np.mean(group_embeddings, axis=0).reshape(1, -1)
        
        # Calculate score and copy images
        for item in items:
            img_emb = np.array(item["embedding"]).reshape(1, -1)
            
            # Math: How close is this image to the perfect center?
            similarity = cosine_similarity(img_emb, centroid)[0][0]
            confidence = round(max(0.0, min(100.0, similarity * 100)), 2)
            
            original_filename = os.path.basename(item["path"])
            # Think User First: Put the score right in the filename!
            new_filename = f"{confidence}%_Match_{original_filename}"
            
            destination = os.path.join(person_dir, new_filename)
            shutil.copy(item["path"], destination)
            print(f"   -> Saved {new_filename}")

# --- THE STARTING LINE ---
if __name__ == "__main__":
    images = clean_and_gather_images(DATASET_FOLDER)
    
    if images:
        embs, valid_paths = extract_face_features(images)
        
        if len(embs) > 0:
            group_and_score_faces(embs, valid_paths)
            print("\n======================================")
            print(f"SUCCESS! All done. Open the '{OUTPUT_FOLDER}' folder to see the results!")
            print("======================================")