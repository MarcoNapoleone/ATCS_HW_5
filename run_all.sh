
# 1) Data prep
python titanic_influence/preprocess.py

# 2) Train model & checkpoints
python titanic_influence/train_model.py

# 3) Compute influences
python titanic_influence/influence_first_order.py --test-indices 0 1 2 --k 20
python titanic_influence/influence_tracin.py      --test-indices 0 1 2 --k 20

# 4) Analyse stability / consistency
python titanic_influence/analysis.py --test-indices 0 1 2 --k 20

# 5) Ablation
python titanic_influence/ablation.py --method first_order --k 20 --test-indices 0 1 2
