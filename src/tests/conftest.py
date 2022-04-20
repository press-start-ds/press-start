import pytest
import pandas as pd


@pytest.fixture(scope="session")
def df_categorical():
    return pd.DataFrame.from_dict(
        {
            "buying": {0: "med", 1: "low", 2: "low", 3: "high", 4: "med"},
            "maint": {0: "med", 1: "med", 2: "low", 3: "med", 4: "low"},
            "year": {0: 2007, 1: 2010, 2: 2015, 3: 1989, 4: 2008},
            "class": {0: "unacc", 1: "good", 2: "acc", 3: "unacc", 4: "unacc"},
        }
    )
