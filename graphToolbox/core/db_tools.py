__author__ = 'Pol Llagostera Blasco'

from sqlalchemy.ext.declarative import declarative_base
import sqlalchemy as db
import pandas as pd
import types

Base = declarative_base()


class TableGraph(Base):
    __tablename__ = 'Graphs'
    g6_hash = db.Column(db.types.String(), primary_key=True, index=True)
    g6 = db.Column(db.types.String(), unique=True, nullable=False)
    #g6_hash = db.Column(db.types.String())
    #g6_id = db.Column(db.types.String(), primary_key=True, index=True)
    nodes = db.Column(db.types.INTEGER)
    edges = db.Column(db.types.INTEGER)
    hamiltonian = db.Column(db.types.BOOLEAN)
    hamiltonian_cycle = db.Column(db.types.String)
    graph_edges = db.Column(db.types.String)
    avg_polynomial = db.Column(db.types.FLOAT)
    polynomial = db.Column(db.types.String)
    spanning_trees = db.Column(db.INTEGER)
    edge_connectivity = db.Column(db.INTEGER)
    min_k2_edge_cuts = db.Column(db.INTEGER)
    automorphisms = db.Column(db.INTEGER)
    diameter = db.Column(db.INTEGER)
    probability_01 = db.Column(db.types.FLOAT)
    probability_02 = db.Column(db.types.FLOAT)
    probability_03 = db.Column(db.types.FLOAT)
    probability_04 = db.Column(db.types.FLOAT)
    probability_05 = db.Column(db.types.FLOAT)
    probability_06 = db.Column(db.types.FLOAT)
    probability_07 = db.Column(db.types.FLOAT)
    probability_08 = db.Column(db.types.FLOAT)
    probability_09 = db.Column(db.types.FLOAT)


class DButilities(object):

    @staticmethod
    def read_table(session, table, column='*', conditions=None):
        """
        Read data from a database and stores it into a dataframe
        :param session: sqlalchemy session
        :param table: table name to read
        :param column: column to read (all by default)
        :param conditions: conditions to make the select (None by default)
        :return: dataframe containing the read data
        """
        if conditions:
            query = "SELECT " + column + " FROM " + table.__tablename__ + " WHERE " + conditions

        else:
            query = "SELECT " + column + " FROM " + table.__tablename__

        df = pd.read_sql_query(query, session.get_bind())

        metadata = db.MetaData()
        graphs = db.Table(table.__tablename__, metadata, autoload=True, autoload_with=session.get_bind())
        table_skeleton = {col.name: col.type.python_type for col in graphs.c}

        df = df.astype(table_skeleton)
        df.set_index(db.inspect(table).primary_key[0].name, inplace=True)
        return df

    @staticmethod
    def create_table(engine, table):
        """
        This method uses the engine object to create all the defined table objects
        and stores the information in metadata.
        :param engine: sqlalchemy engine
        :param table: table to create
        """
        if not engine.dialect.has_table(engine, table.__tablename__):  # If table don't exist, Create.
            table.__table__.create(bind=engine)

    @staticmethod
    def add_column(session, table_name, column_name, column_type):
        """
        Add column to a database table
        :param session: sqlalchemy session
        :param table_name: table name to write
        :param column_name: column to add
        :param column_type: type of the column
        """
        session.execute('ALTER TABLE %s ADD COLUMN %s %s' % (table_name, column_name, column_type))

    @staticmethod
    def bulk_insert(session, df, table_class):
        """
        Insert a bunch of data into a database
        :param session: sqlalchemy session
        :param df: dataframe containing the data to insert
        :param table_class: table type of class
        """
        session.bulk_insert_mappings(table_class, df.to_dict(orient="records"))
        session.commit()

    @staticmethod
    def bulk_update(session, df, table_class):
        """
        Updates a database bunch of data
        :param session: sqlalchemy session
        :param df: dataframe containing the data to update
        :param table_class: table type of class
        """
        session.bulk_update_mappings(table_class, df.to_dict(orient="records"))
        session.commit()

    @staticmethod
    def add_or_update(session, df, table_class):
        """
        Adds data into a database, if the data already exists then is updated.
        :param session: sqlalchemy session
        :param df: dataframe containing the data to add or update
        :param table_class: table type of class
        """
        rows = DButilities.df_to_objects(df)
        primary_key = db.inspect(table_class).primary_key[0].name

        # Create table if not created
        DButilities.create_table(session.get_bind(), table_class)

        data_update = []
        data_insert = []
        for key, values in rows:
            # Returns filter query
            qry = session.query(table_class).filter(getattr(table_class, primary_key) == key)
            # If qry.one(), then returns the filtered object
            tmp = vars(values)
            tmp[primary_key] = key

            # It exists already
            if qry.count():
                # session.merge(element)  # This will update one by one
                data_update.append(vars(values))

            # It doesn't exist yet
            else:
                # session.add(element)  # This will add one by one
                data_insert.append(vars(values))

        # With the filtered data make bulk updates and inserts
        session.bulk_update_mappings(table_class, data_update)
        session.bulk_insert_mappings(table_class, data_insert)
        session.commit()

    @staticmethod
    def df_to_objects(df):
        """
        Transforms a dataframe into a general object
        :param df: dataframe
        :return: general object
        """
        obj_lst = list()
        for key, values in df.to_dict(orient="index").items():
            obj_lst.append((key.decode("utf-8") if type(key) is bytes else key, types.SimpleNamespace(**values)))
        return obj_lst
