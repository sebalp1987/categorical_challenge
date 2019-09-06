from resources.spark import SparkJob
from resources import STRING, functions as f

from pyspark.sql.functions import when, udf
from pyspark.sql.types import StringType, IntegerType

import string

class PreprocessSpark(SparkJob):

    def __init__(self):
        self._spark = self.get_spark_session("preprocess_spark")

    def run(self):
        df = self._extract_data()
        df = self._transform_data(df)
        self._load_data(df)
        self._spark.stop()

    def _extract_data(self):
        df = (self._spark.read.csv(STRING.train, sep=',', header=True, encoding='UTF-8'))

        return df

    @ staticmethod
    def _transform_data(df):
        # delete variables
        df = df.drop(*['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9'])
        cols = df.columns

        # binary variables
        df = df.withColumn('bin_3', when(df['bin_3'] == 'T', 1).otherwise(0))
        df = df.withColumn('bin_4', when(df['bin_4'] == 'Y', 1).otherwise(0))

        # hot encoder
        for col in ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4']:
            types = df.select(col).distinct().collect()
            types = [value[col] for value in types]
            col_new = [when(df[col] == ty, 1).otherwise(0).alias(col + '_' + ty) for ty in types]
            cols = df.columns
            df = df.select(cols + col_new)
            df = df.drop(col)

        # order var:
        '''
        df_corr = df.select(*['ord_3', 'ord_4', 'ord_5', 'target']).toPandas()
        for i in df_corr.columns:
            print(df_corr[i].value_counts())
            print(df_corr.groupby([i, 'target']).size())
        '''

        # ord_1
        dict_ord = {'Novice': '0', 'Contributor': '2', 'Master': '3', 'Grandmaster': '1', 'Expert': '4'}
        funct = udf(lambda x: f.replace_dict(x, dict_ord), StringType())
        df = df.withColumn('ord_1', funct(df['ord_1']))

        # ord 2
        dict_ord = {'Freezing': '0', 'Lava Hot': '1', 'Boiling Hot': '2', 'Cold': '3', 'Hot': '4', 'Warm': '5'}
        funct = udf(lambda x: f.replace_dict(x, dict_ord), StringType())
        df = df.withColumn('ord_2', funct(df['ord_2']))

        # ord 3
        dict_ord = dict(zip(string.ascii_lowercase, range(1, 27, 1)))
        funct = udf(lambda x: f.replace_dict(x, dict_ord), StringType())
        df = df.withColumn('ord_3', funct(df['ord_3']))

        # ord_4
        dict_ord = dict(zip(string.ascii_uppercase, range(1, 27, 1)))
        funct = udf(lambda x: f.replace_dict(x, dict_ord), StringType())
        df = df.withColumn('ord_4', funct(df['ord_4']))

        # ord_5
        dict_ord = df.select('ord_5').distinct().collect()
        dict_ord = [value['ord_5'] for value in dict_ord]
        dict_ord.sort()
        dict_ord = dict(zip(dict_ord, range(1, len(dict_ord) + 1, 1)))
        funct = udf(lambda x: f.replace_dict(x, dict_ord), StringType())
        df = df.withColumn('ord_5', funct(df['ord_5']))

        # grouped day
        df = df.withColumn('weekday', when(df['day'].between(1, 4), 1).otherwise(0))
        df = df.withColumn('weekend', when(df['day'].between(5, 7), 1).otherwise(0))

        # month
        df = df.withColumn('summer', when(df['month'].isin([12, 1, 2]), 1).otherwise(0))
        df = df.withColumn('winter', when(df['month'].isin([6, 7, 8]), 1).otherwise(0))
        df = df.withColumn('autounm', when(df['month'].isin([3, 4, 5]), 1).otherwise(0))
        df = df.withColumn('spring', when(df['month'].isin([9, 10, 11]), 1).otherwise(0))

        return df

    @staticmethod
    def _load_data(df):
        df.coalesce(1).write.mode("overwrite").option("header", "true").option("sep", ",").csv(STRING.train_processed)


if __name__ == "__main__":
    PreprocessSpark().run()
