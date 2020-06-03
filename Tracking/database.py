"""
Database class with POSTGRESQL

Perform sql command:
    CREATE TABLE ~ create_table
    SELECT ~ read
    INSERT TO ~ add
    DELETE ~ delete
    UPSERT ~ upsert_data

"""

import psycopg2


class Database(object):

    # define initial parameters to connect to postgres
    _default = {
        "database": "mydb",
        "user": "postgres",
        "password": "postgres",
        "host": "postgres",
        "port": "5432"
    }

    def __init__(self, **kwargs):
        self.__dict__.update(self._default)
        self.__dict__.update(kwargs)
        # connect to database
        self.db = psycopg2.connect(database=self.database, user=self.user, password=self.password,
                                   host=self.host, port=self.port)
        self.cursor = self.db.cursor()

    def create_table(self, table_name):
        """
        Create a table of analyzing data of a video based upon tracking data
            - Table name: video_name + "_" + video_extention
            - Table items:
                person_id: id assigned to a person in tracking data
                appreance_time: total time of the person in video
                main_activity: the longest time of person's activity video
                        (appear, standing, slow_walking, walking, running)
                time_in_status: activities and its time

        Parameters:
        ----------
            Table name

        Return:
        -------
            create a table in postgres database

        """

        table_items = "(person_id TEXT UNIQUE, " \
                      "appearance_time TEXT, " \
                      "main_activity TEXT, " \
                      "time_in_status TEXT)"

        sql = "CREATE TABLE IF NOT EXISTS {} ".format(table_name)
        sql += table_items

        try:
            self.cursor.execute(sql)
            self.db.commit()
            return True
        except Exception as e:
            print(e)
            return False

    def read(self, table_name, condition=None, condition_value=None):
        """
        Perform postgres command: SELECT * FROM (table_name) WHERE condition = condition_value

        Parameters:
        -----------
            table_name: name of table to perform query command
            condition: condition
            condition_value: condition value

        Return:
        -------
            query data from table_name

        """

        sql = "SELECT * FROM {table}".format(table=table_name)

        if condition is not None:
            qry_condition = " WHERE {condition} = '{condition_value}'".format(condition=condition,
                                                                              condition_value=condition_value)
            sql += qry_condition

        # get all files has performed video tracking in log
        self.cursor.execute(sql)
        qry_result = self.cursor.fetchall()
        # if field_name == "video_name":
        #     result = [d[0] for d in qry_return]
        # else:
        #     result = qry_return

        return qry_result

    def add(self, table_name, **kwargs):
        """
        Perform postgres command: INSERT INTO (table_name) (keys) (values)

        Parameters:
        ----------
            table_name: name of table to insert data
            **kwargs: data to insert to table, format: dictionary of (keys, values)

        Returns:
        --------
            True if successful

        """

        data = kwargs

        keys = ', '.join(data.keys())
        values = ', '.join(['%s'] * len(data))
        sql = 'INSERT INTO {table}({keys}) VALUES ({values})'.format(table=table_name, keys=keys, values=values)
        try:
            self.cursor.execute(sql, tuple(data.values()))
            self.db.commit()
            return True
        except Exception as e:
            print(e)
            self.db.rollback()
            return False

    def delete(self, table_name, condition):
        """
        Perform postgres command: DELETE FROM (table_name) WHERE (condition)

        Parameters:
        -----------
            table_name: name of table to delete rows
            condition: condition to delete

        Return:
        -------
            True if successful

        """

        sql = 'DELETE FROM  {table} WHERE {condition}'.format(table=table_name, condition=condition)
        try:
            self.cursor.execute(sql)
            self.db.commit()
            return True
        except Exception as e:
            print(e)
            self.db.rollback()
            return False

    def upsert_data(self, table_name, condition="video_name", **kwargs):
        """
        Perform UPSERT command of postgres (version > 9.5):
            INSERT INTO (table_name) (keys) VALUES (values) ON CONFLICT (condition) DO UPDATE SET UPDATE ...

        Parameters:
        -----------
            table_name: name of table
            keys, values: keys and values to perform upsert
            condition: condition to perfomr upsert

        Return:
        -------
            True if successful

        """

        data = kwargs
        keys = ", ".join(data.keys())
        values = ", ".join(['%s'] * len(data))

        sql = "INSERT INTO {table} ({keys}) VALUES ({values}) ON CONFLICT ({condition}) DO UPDATE SET".format(
            table=table_name, keys=keys, values=values, condition=condition)

        update = ",".join([" {key} = %s".format(key=key) for key in data])
        sql += update
        try:
            self.cursor.execute(sql, tuple(data.values()) * 2)
            self.db.commit()
            return True
        except Exception as e:
            print(e)
            self.db.rollback()
            return False
