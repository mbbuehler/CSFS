import pandas as pd

def create_0_1(features):
    for f in features:
        print('{}_0'.format(f))
        print('{}_1'.format(f))
        print('{}'.format(f))

# features = ['electricity consumption_[16.0455, 20.243]', 'electricity consumption_(24.87, 29.302]', 'internet users_[6.909, 11.367]', 'exports_(25.0508, 28.424]', 'internet users_(15.631, 19.779]', 'continent_4', 'exports_[13.816, 20.226]', 'oil imports_[0.00995, 8.572]', 'ln_pop_[9.894, 14.0683]', 'oil imports_(12.591, 16.434]','ln_pop_(15.381, 16.16]', 'education expenditures_(5.9, 17.8]', 'education expenditures_(3.133, 4.14]', 'public debt_(71, 235.7]', 'health expenditures_(4.8, 5.9]', 'military expenditures_(3.1, 20.2]', 'gdp growth rate_(6, 121.9]', 'gdp per capita_(8.487, 9.259]', 'region_4', 'ln_pop_(14.0683, 15.381]']
# create_0_1(features)

def create_question_templates(n):
    for i in range(n):
        print('If a country **, what is the chance (or probability) that it won at least one gold medal in the last Olympics? Give your best guess (between 0 and 1).')
        print('If a country **, what is the chance (or probability) that it won at least one gold medal in the last Olympics? Give your best guess (between 0 and 1).')
        print('What is the percentage of countries **? Give your best guess (between 0 and 1).')

def display_questions():
    path = '../datasets/olympia/questions/experiment2/featuresOlympia_hi_lo_10features.csv'
    questions = pd.read_csv(path, header=None)
    for q in questions[1]:
        print(q)
        input()

if __name__ == '__main__':
    display_questions()