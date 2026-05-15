import cv2
import numpy as np
import os
import csv

# ─── 1. VÉRIFICATION NETTETÉ ────────────────────────────
def check_sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if score < 100:
        return False, score, "⚠️ Image floue"
    return True, score, "✅ Image nette"

# ─── 2. VÉRIFICATION VERDURE ────────────────────────────
def check_green_ratio(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    total_pixels = img.shape[0] * img.shape[1]
    green_pixels = np.sum(mask > 0)
    ratio = (green_pixels / total_pixels) * 100
    if ratio < 10:
        return False, ratio, "⚠️ Pas assez de verdure"
    return True, ratio, "✅ Verdure suffisante"

# ─── 3. VÉRIFICATION SUREXPOSITION ──────────────────────
def check_exposure(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = gray.mean()
    if mean_brightness > 240:
        return False, mean_brightness, "⚠️ Image surexposée"
    if mean_brightness < 30:
        return False, mean_brightness, "⚠️ Image trop sombre"
    return True, mean_brightness, "✅ Exposition correcte"

# ─── 4. MODULE QUALITÉ COMPLET ───────────────────────────
def check_image_quality(img):
    print("\n🔍 Vérification qualité image...")
    results = {}

    ok1, score1, msg1 = check_sharpness(img)
    print(f"   Netteté    : {msg1} (score: {score1:.2f})")
    results['nettete'] = {'ok': ok1, 'score': round(score1, 2)}

    ok2, score2, msg2 = check_green_ratio(img)
    print(f"   Verdure    : {msg2} (ratio: {score2:.1f}%)")
    results['verdure'] = {'ok': ok2, 'ratio': round(score2, 1)}

    ok3, score3, msg3 = check_exposure(img)
    print(f"   Exposition : {msg3} (luminosité: {score3:.1f})")
    results['exposition'] = {'ok': ok3, 'luminosite': round(score3, 1)}

    all_ok = ok1 and ok2 and ok3
    if all_ok:
        print("   ✅ Image ACCEPTÉE — qualité suffisante")
    else:
        print("   ❌ Image REJETÉE — qualité insuffisante")

    results['global'] = all_ok
    return results

# ─── TEST SUR 100 IMAGES ─────────────────────────────────
if __name__ == "__main__":
    base = "../data/images/PlantVillage/"
    categories = os.listdir(base)

    acceptees = 0
    rejetees = 0
    total = 0
    csv_data = []

    for cat in categories:
        if total >= 100:
            break
        cat_path = os.path.join(base, cat)
        images = os.listdir(cat_path)
        for img_name in images[:7]:
            if total >= 100:
                break
            img_path = os.path.join(cat_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                print(f"\n{'='*50}")
                print(f"🌿 [{total+1}/100] {cat}")
                result = check_image_quality(img)
                if result['global']:
                    acceptees += 1
                else:
                    rejetees += 1
                total += 1
                csv_data.append({
                    'image': img_name,
                    'categorie': cat,
                    'nettete': result['nettete']['score'],
                    'verdure': result['verdure']['ratio'],
                    'luminosite': result['exposition']['luminosite'],
                    'acceptee': result['global']
                })

    # Sauvegarder en CSV
    os.makedirs("../output", exist_ok=True)
    csv_path = "../output/qualite_100images.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
        writer.writeheader()
        writer.writerows(csv_data)

    print(f"""
╔══════════════════════════════════════╗
║    RÉSULTATS QUALITÉ — {total} IMAGES   ║
╠══════════════════════════════════════╣
║  ✅ Images acceptées : {acceptees:>3}            ║
║  ❌ Images rejetées  : {rejetees:>3}            ║
║  📊 Taux acceptation : {(acceptees/total*100):>5.1f}%         ║
╚══════════════════════════════════════╝
    """)
    print(f"💾 Résultats sauvegardés : {csv_path}")
