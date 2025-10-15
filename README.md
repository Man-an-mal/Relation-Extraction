In this paper, we delve into two RE methodologies: a BiLSTM-based model using Relative Positional Encoding and an SVM-based (BERT + SVM) model. The BiLSTM model captures the sequential dependencies in text. The BERT + SVM model, however, leverages pretrained contextual embeddings to boost relation classification. We train and evaluate both models on the SemEval dataset, applying metrics like accuracy, precision, recall, and F1 score to measure their effectiveness.

Results of Evaluation and Discussion

            F1 Score       Final Accuracy
            
BiLSTM RPE  70.42           75.27
BERT + SVM  59.0            64.26
