DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',  # 替换为你的MySQL用户名
    'password': '123456',  # 替换为你的MySQL密码
    'database': 'tpc_h'  # 替换为你的数据库名
}

# 构建连接字符串
def get_db_connection_string():
    return f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"