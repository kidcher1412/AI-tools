from sqlalchemy import create_engine
from datetime import datetime,timedelta
engine = create_engine('sqlite:///database.db?check_same_thread=False', echo=True)
connection = engine.connect()
from sqlalchemy import Table, Column, Integer, String, MetaData, ForeignKey,insert,update,delete,select,and_,DateTime
metadata = MetaData()
sv = Table('sv',metadata,
            Column('id',String,primary_key=True),
            Column('name',String),
            Column('lop',String),
            Column('sdt',String))
vipham = Table('vipham',metadata,
            Column('id',String),
            Column('name',String),
            Column('lop',String),
            Column('sdt',String),
            Column('thoigian',DateTime,default=datetime.now()),
            Column("path_img",String))

def get_last_time_violation(id,number_violation):
    check = select([vipham]).where(vipham.columns.id == id)
    query = connection.execute(check)
    last_time = query.fetchall()[number_violation-1].thoigian
    return last_time

def get_number_of_violation(id):
    check = select([vipham]).where(vipham.columns.id == id)
    query = connection.execute(check)
    number_of_violation = len(query.fetchall())
    query.close()
    return number_of_violation

def del_all_violation():
    del_all = delete(vipham)
    query = connection.execute(del_all)
    query.close()
def check_true_violation(id,name_img):
    check = select([vipham]).where(vipham.columns.thoigian >= datetime.now()-timedelta(days=0,minutes=3))
    query = connection.execute(check)
    list_violation_two_min = query.fetchall()
    query.close()
    if len(list_violation_two_min) == 0:
        add_violator(id,name_img)
        return True
    list_id = []
    for sv in list_violation_two_min:
        list_id.append(sv[0])
    print(list_id)
    if str(id) in list_id:
        print("tessttttt")
        return False
    add_violator(id, name_img)
    return True
def get_violator_list():
    get_all_violator = select([vipham])
    query = connection.execute(get_all_violator)
    list_violation = query.fetchall()
    query.close()
    return list_violation
def add_violator(id,name_img):
    get_one = select([sv]).where(sv.columns.id == id)
    query = connection.execute(get_one)
    students = query.fetchall()
    query.close()
    if len(students) == 0:
        return False
    student = students[0]
    name_img = "vipham/"+name_img
    add = insert(vipham).values(id=id,name=student[1],lop=student[2],sdt=student[3],path_img = name_img)
    query = connection.execute(add)
    query.close()
    return True


def get_one_student(id):
    get_one = select([sv]).where(sv.columns.id==id)
    query = connection.execute(get_one)
    student = query.fetchall()
    if len(student) == 0:
        return False
    return student[0]
def del_all_students():
    del_all = delete(sv)
    query = connection.execute(del_all)
    query.close()

def get_student():
    get_student = select([sv])
    query = connection.execute(get_student)
    student = query.fetchall()
    query.close()
    return student
def check_student(id):
    check_student = select([sv]).where(sv.columns.id==id)
    query = connection.execute(check_student)
    student_available = query.fetchall()
    query.close()
    if len(student_available) == 0:
        return False
    return True

def add_student(id,name,lop,sdt):
    if check_student(id) == True:
        return False
    new_student = insert(sv).values(id=id,name=name,lop=lop,sdt=sdt)
    query = connection.execute(new_student)
    query.close()
    return True
def del_student(id):
    search_student = delete(sv).where(sv.columns.id ==id)
    query = connection.execute(search_student)
    query.close()
    
def update_student(id,name="",lop="",sdt=""):
    if name == "":
        if lop == "" and sdt == "":
            return False
        elif lop == "" and sdt != "":
            update_student = update(sv).values(sdt=sdt).where(sv.columns.id==id)
            query = connection.execute(update_student)
            query.close()
            return True
        elif lop != "" and sdt == "":
            update_student = update(sv).values(lop=lop).where(sv.columns.id == id)
            query = connection.execute(update_student)
            query.close()
            return True
        else:
            update_student = update(sv).values(lop=lop,sdt=sdt).where(sv.columns.id == id)
            query = connection.execute(update_student)
            query.close()
            return True
    else:
        if lop == "" and sdt == "":
            update_student = update(sv).values(name=name).where(sv.columns.id == id)
            query = connection.execute(update_student)
            query.close()
            return True
        elif lop == "" and sdt != "":
            update_student = update(sv).values(sdt=sdt,name=name).where(sv.columns.id == id)
            query = connection.execute(update_student)
            query.close()
        elif lop != "" and sdt == "":
            update_student = update(sv).values(lop=lop,name=name).where(sv.columns.id == id)
            query = connection.execute(update_student)
            query.close()
            return True
        else:
            update_student = update(sv).values(lop=lop, sdt=sdt,name=name).where(sv.columns.id == id)
            query = connection.execute(update_student)
            query.close()
            return True

