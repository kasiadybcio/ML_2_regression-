import logging
import pandas as pd
logger = logging.getLogger(__name__)

def _cast_data(data :pd.DataFrame, params: dict):
    """
    Cast the data types of the columns in the DataFrame
    """
    mapping_dict = {**{x:'category' for x in params['categorical']},
                        **{x:float for x in params['float']},
                        **{x:int for x in params['integer']},
                        **{x:'datetime64[s]' for x in params['datetime']}}

    cols_to_cast = params['categorical']+params['float']+params['integer']+params['datetime']
    duplicated_cols = set([x for x in cols_to_cast if cols_to_cast.count(x) > 1])

    if duplicated_cols:
        logger.critical("Column type defined more than once: %s",
                        duplicated_cols)

    data = data.astype(mapping_dict)
    missing_cols = set(data.columns) - set(mapping_dict.keys())

    if missing_cols:
        logger.warning("Column type not defined: %s", missing_cols)
    return data
