import cv2
import numpy as np
import os
import csv
from datetime import datetime

# ─── SIMULER LES CONDITIONS D'ÉCLAIRAGE ─────────────────

def simulate_soleil(img):
    """Simule plein soleil : luminosité haute + éblouissement"""
    img_bright = cv2.convertScaleAbs(img, alpha=1.4, beta=80)
    # Ajouter légère surexposition
    hsv = cv2.cvtColor(img_bright, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = np.clip(hsv[:,:,2].astype(int) + 40, 0, 255).astype(np.uint8)
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    print("☀️  Simulation soleil appliquée")
    return result

def simulate_ombre(img):
    """Simule ombre : luminosité basse + faible contraste"""
    img_dark = cv2.convertScaleAbs(img, alpha=0.6, beta=-40)
    # Réduire saturation
    hsv = cv2.cvtColor(img_dark, cv2.COLOR_BGR2HSV)
    hsv[:,:,1] = np.clip(hsv[:,:,1].astype(int) - 30, 0, 255).astype(np.uint8)
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    print("🌥️  Simulation ombre appliquée")
    return result

def simulate_interieur(img):
    """Simule intérieur : légère teinte jaune + luminosité moyenne"""
    img_indoor = cv2.convertScaleAbs(img, alpha=1.0, beta=10)
    # Ajouter légère teinte chaude
    img_indoor[:,:,2] = np.clip(img_indoor[:,:,2].astype(int) + 15, 0, 255).astype(np.uint8)
    img_indoor[:,:,0] = np.clip(img_indoor[:,:,0].astype(int) - 10, 0, 255).astype(np.uint8)
    print("🏠 Simulation intérieur appliquée")
    return img_indoor

# ─── CLAHE ADAPTATIF ────────────────────────────────────
def apply_clahe_adaptive(img, condition):
    params = {
        'soleil':    {'clipLimit': 1.0, 'tileGridSize': (8,8)},
        'ombre':     {'clipLimit': 3.0, 'tileGridSize': (8,8)},
        'interieur': {'clipLimit': 2.0, 'tileGridSize': (8,8)},
    }
    p = params[condition]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=p['clipLimit'], tileGridSize=p['tileGridSize'])
    l_clahe = clahe.apply(l)
    result = cv2.cvtColor(cv2.merge((l_clahe, a, b)), cv2.COLOR_LAB2BGR)
    return result

# ─── GRABCUT ────────────────────────────────────────────
def apply_grabcut(img, condition):
    mask = np.zeros(img.shape[:2], np.uint8)
    bgd = np.zeros((1,65), np.float64)
    fgd = np.zeros((1,65), np.float64)
    h, w = img.shape[:2]
    margin = 20 if condition=='soleil' else 5 if condition=='ombre' else 10
    rect = (margin, margin, w-margin, h-margin)
    cv2.grabCut(img, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    return img * mask2[:,:,np.newaxis]

# ─── ANALYSE QUALITÉ ────────────────────────────────────
def analyze_quality(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    nettete = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = gray.mean()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_vert = cv2.inRange(hsv, np.array([35,40,40]), np.array([85,255,255]))
    verdure = (np.sum(mask_vert>0) / (img.shape[0]*img.shape[1])) * 100
    return round(nettete,2), round(brightness,1), round(verdure,1)

# ─── PIPELINE COMPLET ────────────────────────────────────
def run_simulation_terrain(image_path, condition):
    img_original = cv2.imread(image_path)
    if img_original is None:
        return None

    # 1. Simuler la condition
    if condition == 'soleil':
        img_sim = simulate_soleil(img_original)
    elif condition == 'ombre':
        img_sim = simulate_ombre(img_original)
    else:
        img_sim = simulate_interieur(img_original)

    # 2. Analyser avant traitement
    n_avant, b_avant, v_avant = analyze_quality(img_sim)

    # 3. CLAHE adaptatif
    img_clahe = apply_clahe_adaptive(img_sim, condition)

    # 4. GrabCut
    img_grabcut = apply_grabcut(img_clahe, condition)

    # 5. Analyser après traitement
    n_apres, b_apres, v_apres = analyze_quality(img_grabcut)

    print(f"   Avant  → Netteté: {n_avant:8.2f} | Luminosité: {b_avant:5.1f} | Verdure: {v_avant:5.1f}%")
    print(f"   Après  → Netteté: {n_apres:8.2f} | Luminosité: {b_apres:5.1f} | Verdure: {v_apres:5.1f}%")

    return {
        'image': os.path.basename(image_path),
        'condition': condition,
        'nettete_avant': n_avant,
        'nettete_apres': n_apres,
        'luminosite_avant': b_avant,
        'luminosite_apres': b_apres,
        'verdure_avant': v_avant,
        'verdure_apres': v_apres,
        'amelioration_nettete': round(n_apres - n_avant, 2),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

# ─── TEST SUR 30 IMAGES x 3 CONDITIONS ──────────────────
if __name__ == "__main__":
    base = "../data/images/PlantVillage/"
    categories = os.listdir(base)
    conditions = ['soleil', 'ombre', 'interieur']
    resultats = []

    print("\n🚀 SIMULATION TERRAIN — 3 CONDITIONS")
    print("="*50)

    for condition in conditions:
        print(f"\n\n{'='*50}")
        print(f"  Condition : {condition.upper()}")
        print(f"{'='*50}")
        count = 0
        for cat in categories:
            if count >= 10:
                break
            cat_path = os.path.join(base, cat)
            images = os.listdir(cat_path)
            if images:
                img_path = os.path.join(cat_path, images[0])
                print(f"\n🌿 [{count+1}/10] {cat}")
                res = run_simulation_terrain(img_path, condition)
                if res:
                    resultats.append(res)
                    count += 1

    # Sauvegarder CSV
    os.makedirs("../output", exist_ok=True)
    csv_path = "../output/simulation_terrain.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=resultats[0].keys())
        writer.writeheader()
        writer.writerows(resultats)

    # Statistiques
    print(f"\n\n{'='*50}")
    print("📊 STATISTIQUES PAR CONDITION")
    print("="*50)
    for condition in conditions:
        subset = [r for r in resultats if r['condition'] == condition]
        moy_n_avant = np.mean([r['nettete_avant'] for r in subset])
        moy_n_apres = np.mean([r['nettete_apres'] for r in subset])
        moy_v_avant = np.mean([r['verdure_avant'] for r in subset])
        moy_v_apres = np.mean([r['verdure_apres'] for r in subset])
        amelio = np.mean([r['amelioration_nettete'] for r in subset])

        print(f"\n{'☀️' if condition=='soleil' else '🌥️' if condition=='ombre' else '🏠'}  {condition.upper()}")
        print(f"   Netteté  : {moy_n_avant:.2f} → {moy_n_apres:.2f} (amélioration: {amelio:+.2f})")
        print(f"   Verdure  : {moy_v_avant:.1f}% → {moy_v_apres:.1f}%")

    print(f"\n💾 Résultats : {csv_path}")
    print("✅ Simulation terrain terminée !")
