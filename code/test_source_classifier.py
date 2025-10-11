# -*- coding: utf-8 -*-

"""
test_source_classifier.py
Unit test for svm_source_classifier.py
Author: Jane Heng
Date: Oct.10.2025
"""
import unittest
from svm_source_classifier import load_data, train_model, evaluate_model, run_source_classification
from unittest.mock import patch
import io

class TestSourceClassifier(unittest.TestCase):
    """
    Unit test suite for verifying the functionality of the SVM source classifier pipeline.
    """

    def setUp(self):
        """
        Initializes sample input data and category labels.
        """
        self.categories = ['sci.space', 'rec.sport.hockey', 'talk.politics.misc']
        self.X_sample = [
            "NASA launched a new satellite.",
            "The hockey team won the championship.",
            "Political debates are heating up."
        ]
        self.y_sample = [0, 1, 2]

    def test_load_data(self):
        """
        Tests whether load_data correctly loads training and testing data.
        Check if data lengths match and category names.
        """
        X_train, y_train, X_test, y_test, target_names = load_data(self.categories)
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_test), len(y_test))
        self.assertEqual(len(target_names), len(self.categories))
        self.assertIn('sci.space', target_names)
        print("test_load_data: PASS")

    def test_train_model(self):
        """
        Tests if the model is properly trained on sample input.
        """
        model = train_model(self.X_sample, self.y_sample)
        self.assertTrue(hasattr(model, "predict"))
        print("test_train_model: PASS")

    def test_evaluate_model(self):
        """
        Tests whether evaluate_model function.
        """
        model = train_model(self.X_sample, self.y_sample)
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            evaluate_model(model, self.X_sample, self.y_sample, self.categories)
            output = fake_out.getvalue()
            self.assertIn('sci.space', output)
        print("test_evaluate_model: PASS")

    def test_run_source_classification(self):
        """
        Tests whether run_source_classification function.
        """
        model = run_source_classification(self.categories)
        self.assertTrue(hasattr(model, "predict"))
        print("test_run_source_classification: PASS")

if __name__ == "__main__":
    unittest.main()