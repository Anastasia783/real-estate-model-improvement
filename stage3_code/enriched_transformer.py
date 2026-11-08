import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class FittedEnrichedTransformer(TransformerMixin, BaseEstimator):
    """Объединяет уже обученные sklearn- и autofeat-преобразования"""

    def __init__(
        self,
        sklearn_preprocessor,
        autofeat_imputer,
        autofeat_generator,
        autofeat_columns,
        autofeat_new_columns,
        output_columns,
    ):
        self.sklearn_preprocessor = sklearn_preprocessor
        self.autofeat_imputer = autofeat_imputer
        self.autofeat_generator = autofeat_generator
        self.autofeat_columns = autofeat_columns
        self.autofeat_new_columns = autofeat_new_columns
        self.output_columns = output_columns

    def __sklearn_is_fitted__(self):
        check_is_fitted(self.sklearn_preprocessor)
        check_is_fitted(self.autofeat_imputer)
        return True

    def fit(self, X, y=None):
            raise RuntimeError(
            ""
            )

    def transform(self, X):      
        sklearn_features = pd.DataFrame(
            self.sklearn_preprocessor.transform(X),
            columns=self.sklearn_preprocessor.get_feature_names_out(),
            index=X.index,
        )

        autofeat_input = pd.DataFrame(
            self.autofeat_imputer.transform(
                X.loc[:, self.autofeat_columns]
            ),
            columns=self.autofeat_columns,
            index=X.index,        )

    
        autofeat_features = self.autofeat_generator.transform(
            autofeat_input
        )

        new_features = autofeat_features.loc[
            :, self.autofeat_new_columns
        ].copy()
        new_features.index = X.index
        new_features = new_features.add_prefix("autofeat__")

        enriched = pd.concat(
            [sklearn_features, new_features],
            axis=1,
        ).loc[:, self.output_columns]

        if not np.isfinite(enriched.to_numpy()).all():
            raise ValueError(
                ""
            )

        return enriched

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.output_columns, dtype=object)
