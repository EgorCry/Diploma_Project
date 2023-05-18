import random

import mysql.connector
import time
import torch
import numpy as np
from Sensors.torch_model_class import create_model
from flask import Flask, jsonify, request, redirect, url_for, abort, make_response

app = Flask(__name__)

absolute_id = -1
model = torch.load('Sensors/torch_model.pth')

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


@app.route('/admin', methods=['GET'])
def admin():
    global absolute_id
    global model

    cnx = mysql.connector.connect(**config)
    cursor = cnx.cursor()

    if absolute_id == -1:
        abort(403, 'Forbidden')

    query = 'SELECT Responsible FROM workers WHERE ID_worker = %s'
    cursor.execute(query, (absolute_id,))
    result = cursor.fetchone()

    if not result[0]:
        abort(403, 'Forbidden')

    query = 'SELECT ID_workshop FROM workshop_worker WHERE ID_worker = %s'
    cursor.execute(query, (absolute_id,))
    workshops = cursor.fetchone()[0]

    query = 'SELECT workshop_name FROM workshops WHERE ID_workshop = %s'
    cursor.execute(query, (workshops,))
    workshop_name = cursor.fetchone()[0]

    query = 'SELECT ID_worker FROM workshop_worker WHERE ID_workshop = %s AND ID_worker != %s'
    cursor.execute(query, (workshops, absolute_id,))
    workers = cursor.fetchall()
    worker2 = workers[0][0]
    worker3 = workers[1][0]

    query = 'SELECT r.sensor_value, t.description FROM sensor_readings r ' \
            'INNER JOIN sensors s ON r.ID_sensor = s.ID_sensor ' \
            'INNER JOIN type_sensor t ON s.ID_sensor = t.ID_sensor ' \
            'WHERE s.ID_worker = %s ' \
            'ORDER BY r.time_sensor_reading DESC ' \
            'LIMIT 5'
    cursor.execute(query, (worker2,))
    readings2 = cursor.fetchall()
    print(readings2)
    readings2 = {'temperature': [float(i[0]) for i in readings2 if i[1] == 'Temperature'][0],
                 'pulse': [float(i[0]) for i in readings2 if i[1] == 'Pulse'][0],
                 'high_pressure': [float(i[0]) for i in readings2 if i[1] == 'High Pressure'][0],
                 'low_pressure': [float(i[0]) for i in readings2 if i[1] == 'Low Pressure'][0],
                 'humidity': [float(i[0]) for i in readings2 if i[1] == 'Humidity'][0]}

    input_np = np.array([readings2['temperature'],
                         readings2['pulse'],
                         readings2['high_pressure'],
                         readings2['low_pressure']])

    model = create_model()
    model.eval()

    output_worker2 = round(model(torch.tensor(input_np).float()).detach().item(), 1) * 100
    humidity_worker2 = readings2['humidity'] // 10 * 10
    humidity = random.randint(51, 70) // 10 * 10
    humidity_worker2 = humidity

    if output_worker2 <= 50 and (40 < humidity_worker2 <= 60):
        status_worker2 = 'SAFE'
    elif ((50 < output_worker2 <= 75) and (60 < humidity_worker2 <= 75)) or (
            (0 < output_worker2 <= 50) and (60 < humidity_worker2 <= 75)) or (
            (50 < output_worker2 <= 75) and (40 < humidity_worker2 <= 60)):
        status_worker2 = 'GOOD'
    else:
        status_worker2 = 'BAD'

    query = 'SELECT VALUE FROM device_readings r ' \
            'JOIN device_settings s ON r.ID_device_setting = s.ID_device_setting ' \
            'WHERE s.ID_worker = %s  ' \
            'ORDER BY r.time_value DESC ' \
            'LIMIT 1'
    cursor.execute(query, (worker2,))
    device_worker2 = cursor.fetchone()[0]
    print(device_worker2, worker2)

    query = 'SELECT r.sensor_value, t.description FROM sensor_readings r ' \
            'INNER JOIN sensors s ON r.ID_sensor = s.ID_sensor ' \
            'INNER JOIN type_sensor t ON s.ID_sensor = t.ID_sensor ' \
            'WHERE s.ID_worker = %s ' \
            'ORDER BY r.time_sensor_reading DESC ' \
            'LIMIT 5'
    cursor.execute(query, (worker3,))
    readings3 = cursor.fetchall()
    readings3 = {'temperature': [float(i[0]) for i in readings3 if i[1] == 'Temperature'][0],
                 'pulse': [float(i[0]) for i in readings3 if i[1] == 'Pulse'][0],
                 'high_pressure': [float(i[0]) for i in readings3 if i[1] == 'High Pressure'][0],
                 'low_pressure': [float(i[0]) for i in readings3 if i[1] == 'Low Pressure'][0],
                 'humidity': [float(i[0]) for i in readings3 if i[1] == 'Humidity'][0]}

    input_np = np.array([readings3['temperature'],
                         readings3['pulse'],
                         readings3['high_pressure'],
                         readings3['low_pressure']])

    model = create_model()
    model.eval()

    output_worker3 = round(model(torch.tensor(input_np).float()).detach().item(), 1) * 100
    humidity_worker3 = readings3['humidity'] // 10 * 10
    humidity_worker3 = humidity
    # humidity_worker3 = random.randint(10, 80) // 10 * 10

    if output_worker3 <= 50 and (40 < humidity_worker3 <= 60):
        status_worker3 = 'SAFE'
    elif ((50 < output_worker3 <= 75) and (60 < humidity_worker3 <= 75)) or (
            (0 < output_worker3 <= 50) and (60 < humidity_worker3 <= 75)) or (
            (50 < output_worker3 <= 75) and (40 < humidity_worker3 <= 60)):
        status_worker3 = 'GOOD'
    else:
        status_worker3 = 'BAD'

    device_worker3 = 0

    query = 'SELECT First_name, Last_name FROM workers WHERE ID_worker = %s'
    cursor.execute(query, (absolute_id,))
    result = cursor.fetchone()
    name = result[0]
    surname = result[1]

    # print({'message': f'Hello, Admin!',
    #                 'ID_worker': absolute_id,
    #                 'workshop_name': workshop_name,
    #                 'worker1': worker2,
    #                 'worker2': worker3,
    #                 'first_name': name,
    #                 'second_name': surname,
    #        'prediction_worker2': output_worker2,
    #        'prediction_worker3': output_worker3,
    #        'humidity_worker2': humidity_worker2,
    #        'humidity_worker3': humidity_worker3,
    #        'status_worker2': status_worker2,
    #        'status_worker3': status_worker3,
    #        'worker_2_device': worker_2_device,
    #        'worker_3_device': worker_3_device})

    return jsonify({'message': f'Hello, Admin!',
                    'ID_worker': absolute_id,
                    'workshop_name': workshop_name,
                    'worker1': worker2,
                    'worker2': worker3,
                    'first_name': name,
                    'second_name': surname,
                    'prediction_worker2': output_worker2,
                    'prediction_worker3': output_worker3,
                    'humidity_worker2': humidity_worker2,
                    'humidity_worker3': humidity_worker3,
                    'status_worker2': status_worker2,
                    'status_worker3': status_worker3,
                    'device_worker2': device_worker2,
                    'device_worker3': device_worker3})


@app.route('/worker', methods=['GET', 'POST'])
def worker():
    global absolute_id
    global model

    cnx = mysql.connector.connect(**config)
    cursor = cnx.cursor()

    if absolute_id == -1:
        abort(403, 'Forbidden')

    query = 'SELECT Responsible FROM workers WHERE ID_worker = %s'
    cursor.execute(query, (absolute_id,))
    result = cursor.fetchone()

    if result[0]:
        abort(403, 'Forbidden')

    query = 'SELECT workshop_name FROM workshops WHERE ID_workshop = ' \
            '(SELECT ID_workshop FROM workshop_worker WHERE ID_worker = %s)'
    cursor.execute(query, (absolute_id,))
    workshop_name = cursor.fetchone()[0]

    query = 'SELECT r.sensor_value, t.description FROM sensor_readings r ' \
            'INNER JOIN sensors s ON r.ID_sensor = s.ID_sensor ' \
            'INNER JOIN type_sensor t ON s.ID_sensor = t.ID_sensor ' \
            'WHERE s.ID_worker = %s ' \
            'ORDER BY r.time_sensor_reading DESC ' \
            'LIMIT 5'
    cursor.execute(query, (absolute_id,))
    readings = cursor.fetchall()
    readings = {'temperature': [float(i[0]) for i in readings if i[1] == 'Temperature'][0],
                 'pulse': [float(i[0]) for i in readings if i[1] == 'Pulse'][0],
                 'high_pressure': [float(i[0]) for i in readings if i[1] == 'High Pressure'][0],
                 'low_pressure': [float(i[0]) for i in readings if i[1] == 'Low Pressure'][0],
                 'humidity': [float(i[0]) for i in readings if i[1] == 'Humidity'][0]}

    input_np = np.array([readings['temperature'],
                         readings['pulse'],
                         readings['high_pressure'],
                         readings['low_pressure']])

    model = create_model()
    model.eval()

    output_worker = round(model(torch.tensor(input_np).float()).detach().item(), 1) * 100

    humidity_worker = readings['humidity'] // 10 * 10

    if output_worker <= 50 and (40 < humidity_worker <= 60):
        status_worker = 'SAFE'
    elif ((50 < output_worker <= 75) and (60 < humidity_worker <= 75)) or (
            (0 < output_worker <= 50) and (60 < humidity_worker <= 75)) or (
            (50 < output_worker <= 75) and (40 < humidity_worker <= 60)):
        status_worker = 'GOOD'
    else:
        status_worker = 'BAD'

    query = 'SELECT VALUE FROM device_readings r ' \
            'JOIN device_settings s ON r.ID_device_setting = s.ID_device_setting ' \
            'WHERE s.ID_worker = %s  ' \
            'ORDER BY r.time_value DESC ' \
            'LIMIT 1'
    cursor.execute(query, (absolute_id,))
    device_worker = cursor.fetchone()[0]

    query = 'SELECT First_name, Last_name FROM workers WHERE ID_worker = %s'
    cursor.execute(query, (absolute_id,))
    result = cursor.fetchone()
    name = result[0]
    surname = result[1]

    print({'message': f'Hello, Admin!',
                    'ID_worker': absolute_id,
                    'workshop_name': workshop_name,
                    'first_name': name,
                    'surname': surname,
                    'prediction_worker': output_worker,
                    'humidity_worker': humidity_worker,
                    'status_worker': status_worker,
                    'device_worker': device_worker})

    return jsonify({'message': f'Hello, Admin!',
                    'ID_worker': absolute_id,
                    'workshop_name': workshop_name,
                    'first_name': name,
                    'surname': surname,
                    'prediction_worker': output_worker,
                    'humidity_worker': humidity_worker,
                    'status_worker': status_worker,
                    'device_worker': device_worker})


@app.route('/logout', methods=['GET', 'POST'])
def logout():
    global absolute_id
    if absolute_id == -1:
        return jsonify({'message': 'There is no user in the system'})
    absolute_id = -1
    return jsonify({'message': 'Logged out successfully'})


if __name__ == '__main__':
    app.run()
