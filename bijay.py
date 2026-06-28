import joblib, pickle, os
model_dir = "saved_models"
for fname in os.listdir(model_dir):
    if fname.endswith('.pkl'):
        filepath = os.path.join(model_dir, fname)
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        joblib.dump(model, filepath)
        print(f"✅ Re‑saved: {fname}")