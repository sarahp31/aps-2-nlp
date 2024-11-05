import sqlite3
import bs4

class SQLPostingDatabase():
    name = 'data/careers.db'

    def __init__(self):
        self.conn = sqlite3.connect(self.name, check_same_thread=False)
        self.create_table()

    def create_table(self):
        try:
            table = ''' CREATE TABLE IF NOT EXISTS posting (
                         id INTEGER PRIMARY KEY AUTOINCREMENT,
                         title TEXT NOT NULL,
                         description TEXT,
                         location TEXT NOT NULL,
                         url TEXT NOT NULL,
                         company TEXT NOT NULL,
                         created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                     );'''
            self.conn.execute(table)
            self.conn.commit()
        except Exception as e:
            print('Create Table Exception: ' + str(e))

    def insert(self, name, description, location, url, company):
        description = description if description else ""
        description = bs4.BeautifulSoup(description, "html.parser").get_text()

        if type(location) == list:
            location = ', '.join(location)

        try:
            query = '''INSERT INTO posting (title, description, location, url, company) 
                   VALUES (?, ?, ?, ?, ?)'''
            self.conn.execute(query, (name, description, location, url, company))
            self.conn.commit()
        except Exception as e:
            # print(f"Type of description: {type(location)}, {len(location)}, {location}")
            print('Insert Exception: ' + str(e))

    def get_all(self):
        try:
            cursor = self.conn.cursor()
            query = '''SELECT * FROM POSTING'''
            cursor.execute(query)
            data = []
            for row in cursor.fetchall():
                data.append({})
                data[-1]['title'] = row[1]
                data[-1]['description'] = row[2]
                data[-1]['location'] = row[3]
                data[-1]['url'] = row[4]
                data[-1]['company'] = row[5]
                data[-1]['created_at'] = row[6]
            return data
        except Exception as e:
            print('Get All Exception: ' + str(e))

    def get_company_list(self):
        try:
            cursor = self.conn.cursor()
            query = '''SELECT DISTINCT company FROM POSTING'''
            cursor.execute(query)
            return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            print('Get All Exception: ' + str(e))

    def get_by_company(self, company):
        try:
            cursor = self.conn.cursor()
            query = '''SELECT * FROM POSTING where company = ?'''
            cursor.execute(query, [company])
            data = []
            for row in cursor.fetchall():
                data.append({})
                data[-1]['title'] = row[1]
                data[-1]['description'] = row[2]
                data[-1]['location'] = row[3]
                data[-1]['url'] = row[4]
                data[-1]['company'] = row[5]
                data[-1]['created_at'] = row[6]
            return data
        except Exception as e:
            print('Get By Company Exception: ' + str(e))

    def remove_by_company(self, company):
        try:
            cursor = self.conn.cursor()
            query = '''DELETE FROM POSTING where company = ?'''
            cursor.execute(query, [company])
            return cursor.fetchall()
        except Exception as e:
            print('Remove By Company Exception: ' + str(e))

    def get_count(self):
        try:
            cursor = self.conn.cursor()
            query = '''SELECT COUNT(*) FROM POSTING'''
            cursor.execute(query)
            return cursor.fetchone()[0]
        except Exception as e:
            print('Remove By Company Exception: ' + str(e))

    def search(self, text):
        try:
            cursor = self.conn.cursor()
            query = '''SELECT * FROM POSTING WHERE title like ? or location like ? or company like ? or description like ?'''
            cursor.execute(query, ('%' + text + '%', '%' + text + '%', '%' + text + '%', '%' + text + '%'))
            data = []
            for row in cursor.fetchall():
                data.append({})
                data[-1]['title'] = row[1]
                data[-1]['description'] = row[2]
                data[-1]['location'] = row[3]
                data[-1]['url'] = row[4]
                data[-1]['company'] = row[5]
                data[-1]['created_at'] = row[6]
            return data
        except Exception as e:
            print('Get All Exception: ' + str(e))

    def get_one_by_company(self, company):
        try:
            cursor = self.conn.cursor()
            query = '''SELECT * FROM POSTING where company = ? limit 1'''
            cursor.execute(query, [company])
            data = {}
            row = cursor.fetchone()
            if not row:
                return row
            data['title'] = row[1]
            data['description'] = row[2]
            data['location'] = row[3]
            data['url'] = row[4]
            data['company'] = row[5]
            data['created_at'] = row[6]
            return data
        except Exception as e:
            print('Get By Company Exception: ' + str(e))
