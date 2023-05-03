import mysql.connector
from flask import Flask
from flask_mysqldb import MySQL

app = Flask(__name__)

# параметры подключения к базе данных
config = {
    'user': 'root',
    'password': 'Akatsuki180XBOX',
    'host': '127.0.0.1',
    'port': '3307',
    'database': 'monitor'
}

cnx = mysql.connector.connect(**config)

cursor = cnx.cursor()

query = "SELECT * FROM workers"
cursor.execute(query)

for row in cursor:
    print(row)


if __name__ == '__main__':
    app.run()
