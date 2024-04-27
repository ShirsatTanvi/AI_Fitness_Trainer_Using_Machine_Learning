from flask_mysqldb import MySQL

def configure_db(app):
    # MySQL configurations
    app.config['MYSQL_HOST'] = 'localhost'
    app.config['MYSQL_USER'] = 'root'
    app.config['MYSQL_PASSWORD'] = ''
    app.config['MYSQL_DB'] = 'ai_fitness'
    app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

    # Initialize MySQL
    mysql = MySQL(app)
    return mysql
