import mysql.connector
import random
import datetime


class Sensor:
    def __init__(self, sensor, worker):
        self.sensor_data = {'ID_sensor': sensor, 'ID_worker': worker}

        # параметры подключения к базе данных
        config = {
            'user': 'root',
            'password': 'Akatsuki180XBOX',
            'host': '127.0.0.1',
            'port': '3307',
            'database': 'monitor'
        }

        self.cnx = mysql.connector.connect(**config)
        cursor = self.cnx.cursor()

        query = 'SELECT ID_sensor, Min_value_sensor, Max_value_sensor, Measurements_sensor FROM type_sensor WHERE ID_sensor=%s'
        cursor.execute(query, (sensor,))
        result = cursor.fetchall()

        self.sensor_data['Min_value_sensor'] = float(result[0][1])
        self.sensor_data['Max_value_sensor'] = float(result[0][2])
        self.sensor_data['Measurements_sensor'] = result[0][3]

        print(self.sensor_data)

    def new_value(self, low, high):
        temp = round(random.uniform(low, high), 1)

        if temp < self.sensor_data['Min_value_sensor'] or temp > self.sensor_data['Max_value_sensor']:
            status = 'Bad'
        else:
            status = 'Good'

        query = 'SELECT MAX(ID_sensor_reading) FROM sensor_readings'
        cursor = self.cnx.cursor()
        cursor.execute(query)
        response = cursor.fetchall()
        print(response)
        if response[0][0] is None:
            id = 0
        else:
            id = response[0][0] + 1

        now = datetime.datetime.now()
        formatted_date = now.strftime("%Y-%m-%d %H:%M:%S")
        print("Текущее время: ", formatted_date)

        query = 'INSERT INTO sensor_readings(ID_sensor_reading, time_sensor_reading, Sensor_value, Sensor_value_status, ID_sensor) ' \
                'VALUES (%s, %s, %s, %s, %s)'
        values = (id, formatted_date, temp, status, self.sensor_data['ID_sensor'])
        cursor.execute(query, values)
        self.cnx.commit()
