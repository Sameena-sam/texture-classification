"""
run_pipeline.py
===============
One-command reproduction of the whole experiment:

    python src/run_pipeline.py

Runs, in order:
    1. EDA figures
    2. Feature extraction + caching (prepare)
    3. Feature-space analysis (separability, PCA, t-SNE)
    4. Model training + hyperparameter tuning
    5. Held-out evaluation + comparison table

All artefacts land in results/ and models/.
"""
import time
import eda
import prepare
import feature_analysis
import train
import evaluate


def main():
    t0 = time.time()
    print("\n===== 1/5  EDA =====")
    eda.run_eda()

    print("\n===== 2/5  FEATURE EXTRACTION =====")
    prepare.prepare()

    print("\n===== 3/5  FEATURE ANALYSIS =====")
    feature_analysis.run_feature_analysis()

    print("\n===== 4/5  TRAINING + TUNING =====")
    train.train_supervised()
    train.train_unsupervised()

    print("\n===== 5/5  EVALUATION =====")
    evaluate.evaluate()

    print(f"\nDone in {time.time() - t0:.1f}s. See results/ and models/.")


if __name__ == "__main__":
    main()
