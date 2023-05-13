import mysql.connector
import time
from sensor import Sensor


def main():
    config = {
        'user': 'root',
        'password': 'Akatsuki180XBOX',
        'host': '127.0.0.1',
        'port': '3307',
        'database': 'monitor'
    }

    cnx = mysql.connector.connect(**config)

    query = 'SELECT MAX(ID_sensor_reading) FROM sensor_readings'
    cursor = cnx.cursor()
    cursor.execute(query)
    response = cursor.fetchall()

    if response[0][0] is not None:
        query = 'DELETE FROM sensor_readings'
        cursor.execute(query)
        cnx.commit()

    cnx.close()

    temp1 = Sensor(1, 2)
    temp2 = Sensor(6, 3)

    pulse1 = Sensor(2, 2)
    pulse2 = Sensor(7, 3)

    highp1 = Sensor(3, 2)
    highp2 = Sensor(8, 3)

    lowp1 = Sensor(4, 2)
    lowp2 = Sensor(9, 3)

    hum1 = Sensor(5, 2)
    hum2 = Sensor(10, 3)
    
    i = 0
    
    while i < 4*60*60:
        temp1.new_value(36.6, 36.7)
        pulse1.new_value(88, 90)
        highp1.new_value(115, 119)
        lowp1.new_value(81, 84)
        hum1.new_value(50, 50)

        if i > 120:
            temp2.new_value(37.2, 37.3)
            pulse2.new_value(132, 135)
            highp2.new_value(86, 88)
            lowp2.new_value(59, 61)
            hum2.new_value(50, 50)
        else:
            temp2.new_value(36.9, 37.0)
            pulse2.new_value(100, 110)
            highp2.new_value(109, 111)
            lowp2.new_value(78, 82)
            hum2.new_value(50, 50)

        i += 1
        time.sleep(1)


if __name__ == '__main__':
    main()
