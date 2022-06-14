import pandas as pd
import logging

log = logging.getLogger(__name__)


def load_example_dataset():
    try:
        from nltk.corpus import reuters

        df_reuters = pd.DataFrame(
            [
                (id_, reuters.categories(id_), reuters.raw(id_))
                for id_ in reuters.fileids()
            ],
            columns=["id_doc", "categories", "full_text"],
        )
        return df_reuters

    except ModuleNotFoundError as ex:
        log.error(
            "The nltk package is not installed. Install it with:\n\n" "pip install nltk"
        )
        raise ex
