
# 1) Data prep
python src/preprocess.py

# 2) Train model & checkpoints
python src/train_model.py

# 3) Compute influences
python src/influence_first_order.py --test-indices 0 1 2 --k 20
python src/influence_tracin.py      --test-indices 0 1 2 --k 20

# 4) Analyse stability / consistency
python src/analysis.py --test-indices 0 1 2 --k 20

# 5) Ablation
python src/ablation.py --method first_order --k 20 --test-indices 0 1 2
