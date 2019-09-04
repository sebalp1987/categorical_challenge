
from resources.spark import SparkJob
from resources import STRING

from pyspark.sql.functions import when


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

    def _transform_data(self, df):

        # delete variables
        df = df.drop(*['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9'])
        cols = df.columns

        # binary variables
        df = df.withColumn('bin_3', when(df['bin_3'] == 'T', 1).otherwise(0))
        df = df.withColumn('bin_4', when(df['bin_4'] == 'Y', 1).otherwise(0))

        # hot encoder
        """
        for col in [column for column in df.columns if column != 'id']:
            types = df.select(col).distinct().collect()
            print(types)
        """

        df.show()

        df = df.select(*['id', 'bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4', 'target'])


        return df

    def _load_data(self, df):

        df.coalesce(1).write.mode("overwrite").option("header", "true").option("sep", ",").csv(STRING.train_processed)

if __name__ == "__main__":
    PreprocessSpark().run()