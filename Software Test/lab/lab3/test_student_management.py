# test_student_management.py
import unittest
from student_management import Student, Course, Enrollment


class TestStudentManagement(unittest.TestCase):
    def test_student_creation(self):
        student = Student(1, "张三")
        self.assertEqual(student.student_id, 1)
        self.assertEqual(student.name, "张三")

    def test_course_creation(self):
        course = Course("CS101", "计算机科学导论")
        self.assertEqual(course.course_code, "CS101")
        self.assertEqual(course.course_name, "计算机科学导论")

    def test_enrollment_creation(self):
        student = Student(1, "张三")
        course = Course("CS101", "计算机科学导论")
        enrollment = Enrollment(student, course)
        self.assertEqual(enrollment.student, student)
        self.assertEqual(enrollment.course, course)
        self.assertIsNone(enrollment.grade)  # 成绩默认为None

    def test_set_grade(self):
        student = Student(1, "张三")
        course = Course("CS101", "计算机科学导论")
        enrollment = Enrollment(student, course)
        enrollment.set_grade("A")
        self.assertEqual(enrollment.grade, "A")


if __name__ == "__main__":
    unittest.main()
