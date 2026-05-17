import cv2
import numpy as np
import os
import csv
from datetime import datetime

# ─── PARAMÈTRES CLAHE PAR CONDITION ─────────────────────
CLAHE_PARAMS = {
    'soleil':    {'clipLimit': 1.0, 'tileGridSize': (8, 8)},
    'ombre':     {'clipLimit': 3.0, 'tileGridSize': (8, 8)},
    'interieur': {'clipLimit': 2.0, 'tileGridSize': (8, 8)},
}

# ─── CLAHE ADAPTATIF ────────────────────────────────────
def apply_clahe_adaptive(img, condition='interieur'):
    params = CLAHE_PARAMS.get(condition, CLAHE_PARAMS['interieur'])
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=params['clipLimit'],
        tileGridSize=params['tileGridSize']
    )
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    print(f"✅ CLAHE adaptatif ({condition}) appliqué")
    return result

# ─── GRABCUT ADAPTATIF ──────────────────────────────────
def apply_grabcut_adaptive(img, condition='interieur'):
    mask = np.zeros(img.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    h, w = img.shape[:2]

    # Marge selon condition
    if condition == 'soleil':
        margin = 20
    elif condition == 'ombre':
        margin = 5
    else:
        margin = 10

    rect = (margin, margin, w-margin, h-margin)
    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5,
                cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    result = img * mask2[:,:,np.newaxis]
    print(f"✅ GrabCut adaptatif ({condition}) appliqué")
    return result, mask2

# ─── DÉTECTION CONDITION AUTOMATIQUE ────────────────────
def detect_condition(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = gray.mean()
    std = gray.std()

    if brightness > 180:
        condition = 'soleil'
    elif brightness < 80:
        condition = 'ombre'
    else:
        condition = 'interieur'

    print(f"📊 Luminosité: {brightness:.1f} | Écart-type: {std:.1f}")
    print(f"🌤️  Condition détectée: {condition}")
    return condition, brightness, std

# ─── TEST TERRAIN COMPLET ────────────────────────────────
def test_terrain(image_path, condition_forcee=None):
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Image non trouvée : {image_path}")
        return None

    print(f"\n{'='*50}")
    print(f"🌿 Test terrain : {os.path.basename(image_path)}")
    print(f"{'='*50}")

    # Détecter ou forcer la condition
    if condition_forcee:
        condition = condition_forcee
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = gray.mean()
        std = gray.std()
        print(f"📊 Luminosité: {brightness:.1f} | Condition: {condition}")
    else:
        condition, brightness, std = detect_condition(img)

    # Pipeline adaptatif
    img_clahe = apply_clahe_adaptive(img, condition)
    img_grabcut, mask = apply_grabcut_adaptive(img_clahe, condition)

    # Qualité
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    nettete = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Verdure
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_vert = cv2.inRange(hsv,
        np.array([35, 40, 40]),
        np.array([85, 255, 255]))
    verdure = (np.sum(mask_vert > 0) / (img.shape[0]*img.shape[1])) * 100

    print(f"📊 Netteté: {nettete:.2f} | Verdure: {verdure:.1f}%")
    print(f"✅ Test terrain terminé !")

    return {
        'image': os.path.basename(image_path),
        'condition': condition,
        'luminosite': round(brightness, 1),
        'nettete': round(nettete, 2),
        'verdure': round(verdure, 1),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

# ─── TEST SUR 3 CONDITIONS ───────────────────────────────
if __name__ == "__main__":
    base = "../data/images/PlantVillage/"
    categories = os.listdir(base)
    conditions = ['soleil', 'ombre', 'interieur']
    resultats = []

    print("\n🚀 TEST TERRAIN — 3 CONDITIONS D'ÉCLAIRAGE")
    print("="*50)

    for i, condition in enumerate(conditions):
        print(f"\n\n☀️  CONDITION : {condition.upper()}")
        print("-"*40)
        count = 0
        for cat in categories:
            if count >= 10:
                break
            cat_path = os.path.join(base, cat)
            images = os.listdir(cat_path)
            if images:
                img_path = os.path.join(cat_path, images[0])
                res = test_terrain(img_path, condition_forcee=condition)
                if res:
                    resultats.append(res)
                    count += 1

    # Sauvegarder CSV
    os.makedirs("../output", exist_ok=True)
    csv_path = "../output/test_terrain.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=resultats[0].keys())
        writer.writeheader()
        writer.writerows(resultats)

    # Statistiques par condition
    print(f"\n\n{'='*50}")
    print("📊 STATISTIQUES PAR CONDITION")
    print("="*50)
    for condition in conditions:
        subset = [r for r in resultats if r['condition'] == condition]
        if subset:
            moy_nettete = np.mean([r['nettete'] for r in subset])
            moy_verdure = np.mean([r['verdure'] for r in subset])
            print(f"\n🌤️  {condition.upper()} ({len(subset)} images)")
            print(f"   Netteté moyenne  : {moy_nettete:.2f}")
            print(f"   Verdure moyenne  : {moy_verdure:.1f}%")

    print(f"\n💾 Résultats sauvegardés : {csv_path}")
    print("✅ Test terrain terminé avec succès !")
