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
    features = """marital.status_Married-civ-spouse
relationship_Husband
marital.status_Never-married
education.num_(11, 16]
relationship_Own-child
sex==Male
education.num_(6, 11]
age_(41.333, 65.667]
age_(16.927, 41.333]
hours.per.week_(0.902, 33.667]
native.country_China
native.country_Cuba
native.country_Greece
native.country_Ireland
capital.loss_(2904, 4356]
education_Assoc-acdm
occupation_Armed-Forces
native.country_Hungary
education_Assoc-voc
native.country_Scotland"""
    features = features.split('\n')
    # create_question_templates(len(features))
    create_0_1(features)

    questions = [
        'is not married',
        'is married',
        'are married',
        '',
        '',
        '',
        'has been married or is still married',
        'has never been married',
        'have never been married',
        'has a bad or average education (less education than 67% of the population)',
        'had a very good education (better than 67% of the population)',
        'have a very good education (better than 67% of the population)',
        '',
        '',
        '',
        'is not male',
        'is male',
        'are male', # take out?
        'has a low or high education (either very bad education or very good education, no average education)',
        'has an average education (not very bad and neither very good)',
        'have an average education (not very bad and neither very good)',
        'is younger than 41 years or older than 66 years',
        'is between 41 and 66 years old',
        'are between 41 and 661 years old',
        'is older than 41 years',
        'is younger than 41 years',
        'are younger than 41 years',
        'works more than 34 hours per week',
        'works less than 34 hours per week',
        'work less than 34 hours per week',
        'is not from China',
        'is from China',
        'are from China',
        'is not from Cuba',
        'is from Cuba',
        'are from Cuba',
        'is not from Greece',
        'is from Greece',
        'are from Greece',
        'is not from Ireland',
        'is from Ireland',
        'are from Ireland',
        '',
        '',
        '',
        '',
        '',
        '',
        'does not work for Armed Forces',
        'works for Armed Forces',
        'work for Armed Forces',
        'working in the US is not from Hungary',
        'working in the US is from Hungary',
        'are from Hungary',
        '',
        '',
        '',
        'is not from Scotland',
        'is from Scotland',
        'are from Scotland',
    ]

    for i in range(0, len(questions), 3):
        print('If a person working in the USA *{}*, what is the probability that this person earns more than USD $50K per year?'.format(questions[i]))
        print('If a person working in the USA *{}*, what is the probability that this person earns more than USD $50K per year?'.format(questions[i+1]))
        print('What is the percentage of people working in the USA who *{}*?'.format(questions[i+2]))


    # features = ['electricity consumption per person_[26.757, 1320.325]',
    #             'electricity consumption_(55576666666.667, 3890000000000]',
    #             'region_5']
    # create_question_templates(len(features))

    # display_questions()