import pandas as pd
import pyodbc
import numpy as np


class DataBase:
    def _select(self, query_string: str, connection_string: str) -> pd.DataFrame:
        conn = pyodbc.connect(connection_string)
        cursor = conn.cursor()

        response = cursor.execute(query_string)
        tbl = np.array(response.fetchall())
        try:
            df = pd.DataFrame(tbl, columns=[column[0] for column in response.description])
        except:
            df = pd.DataFrame(columns=[column[0] for column in response.description])
        return df
