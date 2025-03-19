import numpy as np

def predict_price(model, scaler_X, scaler_y, data, seq_length=20):
    # Prepare sequence
    sequence = data[-seq_length:]
    sequence = np.expand_dims(sequence, axis=0)
    
    # Scale features
    scaled_sequence = scaler_X.transform(sequence[0])
    scaled_sequence = np.expand_dims(scaled_sequence, axis=0)
    
    # Predict
    pred_scaled = model.predict(scaled_sequence)
    
    # Inverse scale target
    return scaler_y.inverse_transform(pred_scaled)[0][0]