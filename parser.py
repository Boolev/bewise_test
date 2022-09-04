# Парсер формирует выходные данные в виде словаря
# {
#     'dialog_1': {
#         'greeting_speeches': ['...', '...', '...'],
#         'intro_speeches': ['...', '...', '...'],
#         'name': '...',
#         'company': '...',
#         'end_speeches': ['...', '...', '...'],
#         'has_end_begin': True/False
#     },
#
#     'dialog_2': {
#         'greeting_speeches': ['...', '...', '...'],
#         'intro_speeches': ['...', '...', '...'],
#         'name': '...',
#         'company': '...',
#         'end_speeches': ['...', '...', '...'],
#         'has_end_begin': True/False
#     },
#
#     ...
# }

import pandas as pd
import fasttext
from natasha import Doc, NewsNERTagger, NewsEmbedding, Segmenter

path_to_data = 'insert your path'

# Создание списка имен
with open('names.txt') as file:
    russian_names = file.readlines()
russian_names = [name.replace('\n', '').lower() for name in russian_names]

# Инициализация модулей
segmenter = Segmenter()
emb = NewsEmbedding()
ner_tagger = NewsNERTagger(emb)

# Создание и тренировка модели fasttext
model = fasttext.train_supervised(input="train.txt", epoch=125)


# Классификация текста на классы {'greeting', 'intro', 'end', 'common'}
def classify(text):
    prediction = model.predict(text, k=3)

    if prediction[1][0] > 0.7:
        return prediction[0][0][9:], prediction[1][0]
    else:
        return 'common', prediction[1][0]


# Капитализация имен в тексте для корректной работы модуля извлечения имен Natasha
def up_names(line):
    tokens = line.split()
    for i in range(len(tokens)):
        if tokens[i] in russian_names:
            tokens[i] = tokens[i].title()
    return ' '.join(tokens)


# Извлечение реплик заданного класса
def extract_lines(d_frame, class_name):
    return list(d_frame[d_frame['class'] == class_name]['text'])


# Извлечение заданных сущностей
def extract_entities(text, entity):
    entities = []

    doc = Doc(text)

    doc.segment(segmenter)
    doc.tag_ner(ner_tagger)

    for span in doc.ner.spans:
        if span.type == entity:
            entities.append(text[span.start:span.stop])
  
    return entities


df = pd.read_csv(path_to_data)

# Удаление реплик клиента и перенос всего текста в нижний регистр за исключением имен
df = df[df['role'] == 'manager']
df['text'] = df['text'].apply(lambda x: x.lower()).apply(up_names)

# Добавление столбцов 'class' и 'probability' для возможных служебных целей
df['class'] = [classify(text)[0] for text in df['text']]
df['probability'] = [classify(text)[1] for text in df['text']]


# Формирование результата парсинга в виде словаря
summary = {}

for id in df['dlg_id'].unique():
    local_df = df[df['dlg_id'] == id]

    greeting_speeches = extract_lines(local_df, 'greeting')
    intro_speeches = extract_lines(local_df, 'intro')
    end_speeches = extract_lines(local_df, 'end')
    common_speeches = extract_lines(local_df, 'common')

    has_begin_end = bool(greeting_speeches and end_speeches)


    names = []
    for text in intro_speeches:
        names += extract_entities(text, 'PER')
    for text in greeting_speeches + end_speeches + common_speeches:
        names += extract_entities(text, 'PER')

    name = 'unknown'
    if names:
        name = names[0]


    companies = []
    for text in intro_speeches:
        companies += extract_entities(text, 'ORG')
    for text in greeting_speeches + end_speeches + common_speeches:
        companies += extract_entities(text, 'ORG')

    company = 'unknown'
    if companies:
        company = companies[0]

    summary[f'dialog_{id}'] = {
        'greeting_speeches': greeting_speeches,
        'intro_speeches': intro_speeches,
        'name': name,
        'company': company,
        'end_speeches': end_speeches,
        'has_begin_end': has_begin_end
    }

    print(f'dialog {id + 1} of {len(df["dlg_id"].unique())} processed')