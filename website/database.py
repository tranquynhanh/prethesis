import psycopg2
import psycopg2.extras as extras
import pandas as pd

def select(query, column_names):
    # connect to the PostgreSQL server
    conn = psycopg2.connect(database="Air_quality", user="postgres", password="anh")
        # create a cursor
    cur = conn.cursor()
    cur.execute(query)
    tuples = cur.fetchall()
    cur.close()

    print ("Operation done successfully")
    conn.close()
    df = pd.DataFrame(tuples, columns=column_names)
    return df

def insert(df, table):
    conn = psycopg2.connect(database="Air_quality", user="postgres", password="anh")
    tuples = [tuple(x) for x in df.to_numpy()]
  
    cols = ','.join(list(df.columns))
    # SQL query to execute
    query = "INSERT INTO %s(%s) VALUES %%s" % (table, cols)
    cursor = conn.cursor()
    try:
        extras.execute_values(cursor, query, tuples)
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        conn.rollback()
        cursor.close()
        return 1
    print("the dataframe is inserted")
    cursor.close()
    
   