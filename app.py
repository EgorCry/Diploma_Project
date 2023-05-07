import mysql.connector
from flask import Flask, jsonify, request, redirect, url_for, abort, make_response

app = Flask(__name__)

absolute_id = -1

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
    return jsonify({'Title': 'Main Page', 'ID_worker': absolute_id})

@app.route('/workers')
def get_workers():
    query = "SELECT * FROM workers"
    cursor.execute(query)
    columns = [col[0] for col in cursor.description]
    rows = cursor.fetchall()
    workers = []
    for row in rows:
        worker = {columns[i]: row[i] for i in range(len(columns))}
        workers.append(worker)
    return jsonify(workers)

@app.route('/registration', methods=['POST'])
def registration():
    global absolute_id
    
    if absolute_id != -1:
        return jsonify({'message': 'There is already user in the system'})

    login = request.form['login']
    password = request.form['password']

    query = "SELECT * FROM accounts WHERE Login = %s AND Password = %s"
    cursor.execute(query, (login, password))
    account = cursor.fetchone()

    if account is None:
        # Если пользователь не найден, вернуть сообщение об ошибке
        return jsonify({'error': 'Invalid login or password'}), 401

    absolute_id = account[0]

    query = 'SELECT * FROM workers WHERE ID_worker = %s'
    cursor.execute(query, (absolute_id,))
    result = cursor.fetchone()
    
    # if result[7]:
    #     return redirect(url_for('admin'))
    # return redirect(url_for('worker'))

    if result[7]:
        response = make_response(jsonify({'message': 'admin'}), 200)
        return response
    response = make_response(jsonify({'message': 'worker'}), 200)
    return response


@app.route('/admin', methods=['GET', 'POST'])
def admin():
    global absolute_id

    if absolute_id == -1:
        abort(403, 'Forbidden')

    query = 'SELECT Responsible FROM workers WHERE ID_worker = %s'
    cursor.execute(query, (absolute_id,))
    result = cursor.fetchone()

    if not result[0]:
        abort(403, 'Forbidden')

    if request.method == 'POST' and request.form.get('action') == 'exit':
        return redirect(url_for('logout'))
    else:
        return jsonify({'message': 'Hello, Admin!', 'ID_worker': absolute_id})


@app.route('/worker', methods=['GET', 'POST'])
def worker():
    global absolute_id

    if absolute_id == -1:
        abort(403, 'Forbidden')

    query = 'SELECT Responsible FROM workers WHERE ID_worker = %s'
    cursor.execute(query, (absolute_id,))
    result = cursor.fetchone()

    if result[0]:
        abort(403, 'Forbidden')

    if request.method == 'POST' and request.form.get('action') == 'exit':
        return redirect(url_for('logout'))
    else:
        return jsonify({'message': 'Hello, Admin!', 'ID_worker': absolute_id})


@app.route('/logout', methods=['GET', 'POST'])
def logout():
    global absolute_id
    if absolute_id == -1:
        return jsonify({'message': 'There is no user in the system'})
    absolute_id = -1
    return jsonify({'message': 'Logged out successfully'})


if __name__ == '__main__':
    app.run()
