import mysql.connector
from flask import Flask, jsonify, request, redirect, url_for

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

@app.route('/')
def main():
    return jsonify({'Title': 'Main Page'})

@app.route('/workers')
def get_workers():
    query = "SELECT * FROM workers"
    cursor.execute(query)
    rows = cursor.fetchall()
    return jsonify(rows)

@app.route('/registration', methods=['POST'])
def registration():
    login = request.form['login']
    password = request.form['password']

    query = "SELECT * FROM accounts WHERE Login = %s AND Password = %s"
    cursor.execute(query, (login, password))
    account = cursor.fetchone()

    if account is None:
        # Если пользователь не найден, вернуть сообщение об ошибке
        return jsonify({'error': 'Invalid login or password'}), 401

    query = 'SELECT * FROM workers WHERE ID_worker = %s'
    id = account[0]
    cursor.execute(query, (id,))
    result = cursor.fetchone()
    
    if result[5] == 'Admin':
        return redirect(url_for('admin'))
    return redirect(url_for('worker'))


@app.route('/admin')
def admin():
    return jsonify({'message': 'Hello, Admin!'})


@app.route('/worker')
def worker():
    return jsonify({'message': 'Hello, Worker!'})


if __name__ == '__main__':
    app.run()
