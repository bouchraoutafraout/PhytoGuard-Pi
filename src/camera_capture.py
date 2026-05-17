import cv2
import numpy as np
import os
import sys
import csv

def capture_image(image_path):
    img = cv2.imread(image_path)
    if img is not None:
        print(f"✅ Image simulée : {os.path.basename(image_path)}")
    return img

def run_capture_pipeline_batch(nb_images=100):
    print("\n" + "="*50)
    print(f"🌿 PhytoGuard — Test sur {nb_images} images")
    print("="*50)

    sys.path.append(os.path.dirname(__file__))
    from quality_module import check_image_quality
    from pipeline_cv import apply_clahe, apply_grabcut
    from pipeline_cv import extract_hsv_features, extract_lbp_features

    base = "../data/images/PlantVillage/"
    categories = os.listdir(base)

    resultats = []
    acceptees = 0
    rejetees = 0
    total = 0

    for cat in categories:
        if total >= nb_images:
            break
        cat_path = os.path.join(base, cat)
        images = os.listdir(cat_path)
        for img_name in images[:7]:
            if total >= nb_images:
                break
            img_path = os.path.join(cat_path, img_name)
            img = capture_image(img_path)
            if img is None:
                continue

            print(f"\n[{total+1}/{nb_images}] 🌿 {cat}")

            # Vérifier qualité
            quality = check_image_quality(img)

            if not quality['global']:
                print("❌ Image rejetée")
                rejetees += 1
                total += 1
                continue

            # Pipeline CV
            img_clahe = apply_clahe(img)
            img_grabcut, mask = apply_grabcut(img_clahe)
            hsv = extract_hsv_features(img_grabcut)
            lbp = extract_lbp_features(img_grabcut)

            acceptees += 1
            total += 1

            resultats.append({
                'categorie': cat,
                'image': img_name,
                'nettete': quality['nettete']['score'],
                'verdure': quality['verdure']['ratio'],
                'luminosite': quality['exposition']['luminosite'],
                'hsv_h': round(hsv[0], 2),
                'hsv_s': round(hsv[2], 2),
                'hsv_v': round(hsv[4], 2),
                'lbp_max': round(float(np.max(lbp)), 3),
                'acceptee': quality['global']
            })

    # Sauvegarder en CSV
    os.makedirs("../output", exist_ok=True)
    csv_path = "../output/capture_pipeline_100.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=resultats[0].keys())
        writer.writeheader()
        writer.writerows(resultats)

    print(f"""
╔══════════════════════════════════════╗
║   RÉSULTATS — {total} IMAGES TESTÉES   ║
╠══════════════════════════════════════╣
║  ✅ Images acceptées : {acceptees:>3}            ║
║  ❌ Images rejetées  : {rejetees:>3}            ║
║  📊 Taux acceptation : {(acceptees/total*100):>5.1f}%         ║
╚══════════════════════════════════════╝
    """)
    print(f"💾 Résultats sauvegardés : {csv_path}")
    print("✅ Pipeline complet terminé avec succès !")

if __name__ == "__main__":
    run_capture_pipeline_batch(nb_images=100)
