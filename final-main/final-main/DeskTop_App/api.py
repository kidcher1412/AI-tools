from flask import Flask,request,jsonify
from sqlalchemy import create_engine
from datetime import datetime,timedelta
from database import get_number_of_violation,get_one_student,get_student,get_number_of_violation,get_violator_list,get_last_time_violation
app = Flask(__name__)


@app.route('/check',methods=['GET'])
def return_in4():
    id_student = request.args.get("id")

    print(get_violator_list())
    NumberViolation = get_number_of_violation(str(id_student))
    if int(NumberViolation) == 0:
        data_return = {
            "Violation":False,
            "data":""}
        return jsonify(data_return)
    else:
        LastTime = str(get_last_time_violation(id_student,int(NumberViolation)))
    in4 = get_one_student(str(id_student))
    data = {
        "IdStudent":in4[0],
        "NameStudent":in4[1],
        "Class":in4[2],
        "PhoneNumber":in4[3],
        "NumberViolation":NumberViolation,
        "LastTime":LastTime
    }
    data_return = {
        "Violation":True,
        "data":data
    }
    return jsonify(data_return)

if __name__ == '__main__':
    app.run(debug=True,port=9999)