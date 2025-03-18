import unittest
from app.models import build_model

class TestModels(unittest.TestCase):
    def test_build_model(self):
        seq_length = 20
        num_features = 11
        model = build_model(seq_length, num_features, lstm_units=128, gru_units=64, dense_units=64, dropout_rate=0.15, learning_rate=0.001)
        self.assertIsNotNone(model)
        self.assertEqual(model.input_shape, (None, seq_length, num_features))
        self.assertEqual(model.output_shape[-1], 1)

if __name__ == "__main__":
    unittest.main()
