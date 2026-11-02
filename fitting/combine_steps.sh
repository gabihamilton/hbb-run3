cd results/25Nov11_v14_private/combination

# 1. Build Workspace
text2workspace.py full_run3_combined.txt \
    -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel \
    --PO verbose \
    --PO 'map=.*/Zgammabb:r[1,-20,20]' \
    -o workspace_run3.root

# 2. Significance
echo "--- Calculating Significance ---"
combine -M Significance workspace_run3.root --expectSignal 1 --rMin -20 --rMax 20

# 3. Best Fit Value
echo "--- Measuring Best Fit r ---"
combine -M MultiDimFit workspace_run3.root --expectSignal 1 \
    --rMin -100 --rMax 100 --algo singles --robustFit 1 --cminDefaultMinimizerStrategy 0

# 4. Extract Shapes (for Plotting)
echo "--- Extracting Post-Fit Shapes ---"
combine -M FitDiagnostics workspace_run3.root --expectSignal 1 \
    --rMin -20 --rMax 20 \
    --saveShapes --saveWithUncertainties \
    --robustFit=1
