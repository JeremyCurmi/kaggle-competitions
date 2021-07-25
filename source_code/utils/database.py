import pandas as pd
from tqdm import tqdm


def insert_with_progressbar(df: pd.DataFrame, table_name: str, sql_engine, batches: int = 10):
    def batch_processor(seq, size):
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))

    batch_size = int(len(df) / batches)
    with tqdm(total=len(df)) as pbar:
        for i, cdf in enumerate(batch_processor(df, batch_size)):
            replace = "replace" if i == 0 else "append"
            cdf.to_sql(con=sql_engine, name=table_name, if_exists='append', index=False)
            pbar.update(batch_size)
