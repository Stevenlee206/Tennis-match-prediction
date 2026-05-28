from copy import deepcopy
from collections import deque

import numpy as np
from hmmlearn.hmm import GaussianHMM


class HMMClassifier:
    def __init__(
        self,
        n_components=3,
        covariance_type="diag",
        n_iter=100,
        tol=1e-2,
        min_covar=1e-3,
        sequence_length=5,
        random_state=42,
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol
        self.min_covar = min_covar
        self.sequence_length = sequence_length
        self.random_state = random_state
        self.classes_ = None
        self.models_ = {}
        self.majority_class_ = None

    def _build_model(self, n_components, covariance_type):
        return GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=self.n_iter,
            tol=self.tol,
            min_covar=self.min_covar,
            random_state=self.random_state,
            verbose=False,
        )

    def _effective_sequence_length(self, X):
        sequence_length = int(getattr(self, "sequence_length", 1))
        if sequence_length < 1:
            sequence_length = 1
        return max(1, min(sequence_length, len(X)))

    def _build_training_sequences(self, X, y, class_label, sequence_length):
        class_indices = np.where(y == class_label)[0]
        if len(class_indices) == 0:
            return np.empty((0, X.shape[1]), dtype=X.dtype), []

        lengths = []
        spans = []
        total = 0

        for index in class_indices:
            start_index = max(0, index - sequence_length + 1)
            length = index - start_index + 1
            if length <= 0:
                continue
            lengths.append(length)
            spans.append((start_index, index + 1))
            total += length

        if not lengths:
            return np.empty((0, X.shape[1]), dtype=X.dtype), []

        # preallocate flattened class observations to avoid many small arrays
        X_class = np.empty((total, X.shape[1]), dtype=X.dtype)
        pos = 0
        for (s, e) in spans:
            seg = X[s:e]
            X_class[pos : pos + len(seg)] = seg
            pos += len(seg)

        return X_class, lengths

    def _score_sequence(self, model, sequence):
        try:
            return float(model.score(sequence))
        except Exception:
            return -1e12

    def _build_prediction_sequences(self, X, context=None):
        X = np.asarray(X)
        if context is None:
            context = np.empty((0, X.shape[1]), dtype=float)
        else:
            context = np.asarray(context)
            if context.ndim == 1:
                context = context.reshape(1, -1)

        history = context.copy()
        sequences = []

        for row in X:
            row = np.asarray(row).reshape(1, -1)
            history = np.vstack([history, row]) if history.size else row
            effective_length = self._effective_sequence_length(history)
            sequences.append(history[-effective_length:])

        return sequences

    def fit(self, X, y):
        classes, majority_class, models = self._train_class_models(X, y, n_iter=self.n_iter)
        self.classes_ = classes
        self.majority_class_ = majority_class
        self.models_ = models
        return self

    def _train_class_models(self, X, y, n_iter=None, warm_start_models=None):
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array for HMMClassifier.")

        classes, counts = np.unique(y, return_counts=True)
        if len(classes) == 0:
            raise ValueError("Cannot fit HMMClassifier with no labels.")

        majority_class = classes[int(np.argmax(counts))]
        model_iter = self.n_iter if n_iter is None else n_iter
        models = deepcopy(warm_start_models) if warm_start_models else {}
        sequence_length = self._effective_sequence_length(X)

        if len(classes) == 1:
            return classes, majority_class, models

        for class_label in classes:
            X_class, lengths = self._build_training_sequences(X, y, class_label, sequence_length)

            if X_class.size == 0 or not lengths:
                continue

            total_observations = len(X_class)
            n_components = max(1, min(self.n_components, total_observations))
            model = models.get(class_label)

            if model is None:
                model = self._build_model(n_components, self.covariance_type)
            else:
                model.n_iter = model_iter
                model.init_params = ""

            model.n_iter = model_iter

            try:
                model.fit(X_class, lengths=lengths)
            except Exception:
                try:
                    model = self._build_model(1, "diag")
                    model.n_iter = model_iter
                    model.fit(X_class, lengths=lengths)
                except Exception:
                    continue

            models[class_label] = model

        return classes, majority_class, models

    def _log_likelihood_matrix(self, X, context=None):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array for prediction.")

        if self.classes_ is None:
            raise ValueError("HMMClassifier must be fitted before prediction.")

        if len(self.models_) < 2:
            return np.zeros((X.shape[0], len(self.classes_)))

        scores = np.full((X.shape[0], len(self.classes_)), -np.inf, dtype=float)

        # Use a deque limited to sequence_length to avoid growing full-history arrays
        seq_len = int(getattr(self, 'sequence_length', 1))
        if seq_len < 1:
            seq_len = 1

        if context is None:
            context = np.empty((0, X.shape[1]), dtype=X.dtype)
        else:
            context = np.asarray(context)
            if context.ndim == 1:
                context = context.reshape(1, -1)

        # initialize deque with last up to seq_len rows from context
        history_deque = deque(maxlen=seq_len)
        if context.size:
            for r in context[-seq_len:]:
                history_deque.append(np.asarray(r))

        for row_idx, row in enumerate(X):
            history_deque.append(np.asarray(row))
            try:
                sequence = np.vstack(history_deque)
            except Exception:
                sequence = np.asarray(list(history_deque))

            for class_idx, class_label in enumerate(self.classes_):
                model = self.models_.get(class_label)
                if model is None:
                    continue
                scores[row_idx, class_idx] = self._score_sequence(model, sequence)

        return scores

    def predict_proba(self, X, context=None):
        X = np.asarray(X)

        if self.classes_ is None:
            raise ValueError("HMMClassifier must be fitted before prediction.")

        if len(self.models_) < 2:
            proba = np.zeros((X.shape[0], len(self.classes_)), dtype=float)
            if self.majority_class_ is not None:
                class_idx = int(np.where(self.classes_ == self.majority_class_)[0][0])
                proba[:, class_idx] = 1.0
            return proba

        log_likelihoods = self._log_likelihood_matrix(X, context=context)
        max_logits = np.max(log_likelihoods, axis=1, keepdims=True)
        stabilized = np.exp(log_likelihoods - max_logits)
        denom = np.sum(stabilized, axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        return stabilized / denom

    def predict(self, X, context=None):
        if self.classes_ is None:
            raise ValueError("HMMClassifier must be fitted before prediction.")

        proba = self.predict_proba(X, context=context)
        return self.classes_[np.argmax(proba, axis=1)]
