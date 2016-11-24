import pandas as pd

def create_0_1(features):
    for f in features:
        print('{}_0'.format(f))
        print('{}_1'.format(f))
        print('{}'.format(f))

# features = ['electricity consumption_[16.0455, 20.243]', 'electricity consumption_(24.87, 29.302]', 'internet users_[6.909, 11.367]', 'exports_(25.0508, 28.424]', 'internet users_(15.631, 19.779]', 'continent_4', 'exports_[13.816, 20.226]', 'oil imports_[0.00995, 8.572]', 'ln_pop_[9.894, 14.0683]', 'oil imports_(12.591, 16.434]','ln_pop_(15.381, 16.16]', 'education expenditures_(5.9, 17.8]', 'education expenditures_(3.133, 4.14]', 'public debt_(71, 235.7]', 'health expenditures_(4.8, 5.9]', 'military expenditures_(3.1, 20.2]', 'gdp growth rate_(6, 121.9]', 'gdp per capita_(8.487, 9.259]', 'region_4', 'ln_pop_(14.0683, 15.381]']
# create_0_1(features)

# def create_question_templates(n):
    # for i in range(n):
    #     print('If a country **, what is the chance (or probability) that it won at least one gold medal in the last Olympics? Give your best guess (between 0 and 1).')
    #     print('If a country **, what is the chance (or probability) that it won at least one gold medal in the last Olympics? Give your best guess (between 0 and 1).')
    #     print('What is the percentage of countries **? Give your best guess (between 0 and 1).')
    #
    # for i in range(n):
    #     print('If a secondary education *student *, what is the probability that the student’s final grade is in the better half of the class?')
    #     print('If a secondary education *student *, what is the chance (or probability) that the student scored the same or better than the weaker half of the class? Give your best guess (between 0 and 1).')
    #     print('What is the percentage of secondary education students **? Give your best guess (between 0 and 1).')

def display_questions():
    path = '../datasets/student/questions/experiment2_por/questions.csv'
    questions = pd.read_csv(path, header=None)
    for q in questions[1]:
        print(q)
        input()

if __name__ == '__main__':
    # display_questions()
    # todo: create questions for student
    features = ['Mjob_other', 'higher==yes', 'absences_(21.333, 32]', 'failures_(-0.003, 1]', 'paid==yes', 'Fjob_teacher', 'Medu_(-0.004, 1.333]', 'health_(2.333, 3.667]', 'failures_(2, 3]', 'famsize==LE3', 'age_(19.667, 22]', 'studytime_(0.997, 2]', 'Pstatus==T', 'failures_(1, 2]', 'Mjob_at_home']
    # create_question_templates(len(features))
    # create_0_1(features)

    questions = ["'s mother works in teaching, health care, civil services (e.g. administrative or police) or at home",
                "'s mother works in an other field than teaching, health care, civil services (e.g. administrative or police) or at home",
                 "whose mothers work in an other field than teaching, health care, civil services (e.g. administrative or police) or at home",
                 ' does not want to take higher education',
                 ' wants to take higher education',
                 'willing to take higher education',
                 ' has few or a medium number of absences at school',
                 ' has many absences at school',
                 'having many absences at school',
                 ' has failed more than one class',
                 ' has failed a maximum of one class',
                 'having failed a maximum of one class',
                 ' has taken no extra paid lessons',
                 ' has taken extra paid lessons',
                 'having taken extra paid lessons',
                 "'s father is not a teacher",
                 "'s father is a teacher",
                 'whose father is a teacher',
                 "'s mother had more than Primary education until 4th grade",
                 "'s mother had Primary education until 4th grade or less",
                 "whose mothers had Primary education until 4th grade or less",
                 "'s health status is either high or low",
                 "'s health status is average",
                 "whose health status is average",
                 " has failed less than three classes",
                 " has failed three classes or more",
                 "having failed three classes or more",
                 "'s family size is greater than three",
                 "'s family size is less or equal to three",
                 "with a family size less or equal to three",
                 " belongs to the younger or middle-aged part of the class (19 years or younger)",
                 " belongs to the older part of the class (older than 19 years)",
                 "belonging to the older part of the class (older than 19 years)",
                 " studies at least an average time (more than 5 hours per week)",
                 " studies little (5 hours per week or less)",
                 "studying little (5 hours per week or less)",
                 "'s parents live apart",
                 "'s parents live together",
                 "whose parents live together",
                 " has failed either fewer classes than average or more classes than average (less than 2 or more than 2 classes)",
                 " has failed an average number of classes (two classes)",
                 "having failed an average number of classes (two classes)",
                 "'s mother does not work at home",
                 "'s mother works at home",
                 "whose mothers work at home",

                 ]

    for i in range(0, len(questions), 3):
        print('If a secondary education *student{}*, what is the probability that the student’s final grade is in the better half of the class?'.format(questions[i]))
        print('If a secondary education *student{}*, what is the probability that the student’s final grade is in the better half of the class?'.format(questions[i+1]))
        print('What is the percentage of secondary education students *{}*?'.format(questions[i+2]))


    # features = ['electricity consumption per person_[26.757, 1320.325]',
    #             'electricity consumption_(55576666666.667, 3890000000000]',
    #             'region_5']
    # create_question_templates(len(features))

    # display_questions()