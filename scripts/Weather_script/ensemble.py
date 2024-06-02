from sklearn.neural_network import MLPRegressor
import numpy as np

features = np.column_stack((pred_ml, pred_nwp1, pred_nwp2))
target = actual_rainfall

meta_model = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=42)
meta_model.fit(features, target)

# Now, use the meta-model to make combined predictions
combined_predictions = meta_model.predict(features)
