import sqlite3 as sql

con = sql.connect('dataset.db')
query = """
SELECT nickname FROM employers;
"""
data = con.execute(query)
print('There are all nicknames in dataset.db')
for row in data:
    print(row[0])
con.close()
