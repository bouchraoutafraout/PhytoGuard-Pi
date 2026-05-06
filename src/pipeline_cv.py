import cv2
import numpy as np
import os
import csv
from skimage.feature import local_binary_pattern

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        print(f"Erreur : impossible de charger {path}")
        return None
    print(f"✅ Image chargée : {img.shape}")
    return img

def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    print("✅ CLAHE appliqué")
    return result

def apply_grabcut(img):
    mask = np.zeros(img.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    h, w = img.shape[:2]
    rect = (10, 10, w-10, h-10)
    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5,
                cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    result = img * mask2[:,:,np.newaxis]
    print("✅ GrabCut appliqué")
    return result, mask2

def check_quality(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(f"📊 Score netteté : {score:.2f}")
    if score < 100:
        print("⚠️  Image floue !")
        return False, score
    print("✅ Image nette")
    return True, score

def extract_hsv_features(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    features = []
    for i in range(3):
        features.append(hsv[:,:,i].mean())
        features.append(hsv[:,:,i].std())
    print(f"✅ Features HSV : {np.round(features, 2)}")
    return np.array(features)

def extract_lbp_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0,10))
    hist = hist / hist.sum()
    print(f"✅ Features LBP : {np.round(hist, 3)}")
    return hist

def run_pipeline(image_path, category):
    print(f"\n{'='*50}")
    print(f"🌿 Catégorie : {category}")
    print(f"📄 Fichier   : {os.path.basename(image_path)}")
    print(f"{'='*50}")
    img = load_image(image_path)
    if img is None:
        return None
    nette, score = check_quality(img)
    img_clahe = apply_clahe(img)
    img_grabcut, mask = apply_grabcut(img_clahe)
    hsv = extract_hsv_features(img_grabcut)
    lbp = extract_lbp_features(img_grabcut)
    print(f"✅ Pipeline terminé !")
    return {
        "categorie": category,
        "fichier": os.path.basename(image_path),
        "nette": nette,
        "score_nettete": round(score, 2),
        "hsv_h_moy": round(hsv[0], 2),
        "hsv_s_moy": round(hsv[2], 2),
        "hsv_v_moy": round(hsv[4], 2),
        "lbp_dominant": round(float(np.max(lbp)), 3),
    }

if __name__ == "__main__":
    base = "../data/images/PlantVillage/"
    categories = os.listdir(base)
    resultats = []
    images_ok = 0
    images_floues = 0

    # ── Tester 20 images ──────────────────────────────
    for cat in categories:
        cat_path = os.path.join(base, cat)
        images = os.listdir(cat_path)
        if images and len(resultats) < 20:
            img_path = os.path.join(cat_path, images[0])
            res = run_pipeline(img_path, cat)
            if res:
                resultats.append(res)
                if res["nette"]:
                    images_ok += 1
                else:
                    images_floues += 1

    # ── Sauvegarder en CSV ────────────────────────────
    csv_path = "../output/resultats.csv"
    os.makedirs("../output", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=resultats[0].keys())
        writer.writeheader()
        writer.writerows(resultats)
    print(f"\n💾 Résultats sauvegardés dans : {csv_path}")

    # ── Statistiques globales ─────────────────────────
    scores = [r["score_nettete"] for r in resultats]
    print(f"""
╔══════════════════════════════════════╗
║        STATISTIQUES GLOBALES        ║
╠══════════════════════════════════════╣
║  Images analysées   : {len(resultats):>3}            ║
║  Images nettes      : {images_ok:>3}            ║
║  Images floues      : {images_floues:>3}            ║
║  Score netteté min  : {min(scores):>8.2f}       ║
║  Score netteté max  : {max(scores):>8.2f}       ║
║  Score netteté moy  : {np.mean(scores):>8.2f}       ║
╚══════════════════════════════════════╝
    """)
    print(f"🎉 Pipeline Semaine 1 terminé avec succès !")
