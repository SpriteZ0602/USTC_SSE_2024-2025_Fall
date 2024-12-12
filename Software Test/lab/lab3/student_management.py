class Student:
    def __init__(self, student_id, name):
        self.student_id = student_id
        self.name = name


class Course:
    def __init__(self, course_code, course_name):
        self.course_code = course_code
        self.course_name = course_name


class Enrollment:
    def __init__(self, student, course, grade=None):
        self.student = student
        self.course = course
        self.grade = grade

    def set_grade(self, grade):
        self.grade = grade
