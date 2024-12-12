import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from arfc.linear_model import LinearARModel
from arfc.multi import MultiARTransformer


def test_multi_ar_transformer(random_group_data: pd.DataFrame):
    lin = LinearRegression(fit_intercept=False)
    model = LinearARModel(lin, order=3, lag=2)
    multi_transformer = MultiARTransformer(model)

    armats = multi_transformer.fit_transform(random_group_data)
    groups = random_group_data.iloc[:, 0].values
    assert len(armats) == len(np.unique(groups))
    assert armats.shape == (3, 3, 64, 64)

    pred = multi_transformer.predict(random_group_data)
    assert pred.shape == random_group_data.shape

    score = multi_transformer.score(random_group_data)
    assert score > 0
