import streamlit as st 
# from annotated_text import annotated_text
import spacy 

st.title('Text Anonymizer in many languages')

def lang_models():
    en = spacy.load('en_core_web_sm')
    # fr = spacy.load('fr_core_news_sm')
    spn = spacy.load('es_core_news_sm')
    # jpn = spacy.load('jp_core_news_sm')

    models = {'English':en,
            'Spanish':spn}
    
    return models
    
def process_data(doc,entity,anonymize=False):
    tokens = []

    for token in doc:
        if (token.ent_type_ == 'PERSON')&('PER' in entity):
            tokens.append((token.text,'Person','#faa'))
        
        elif (token.ent_type_ in ['GPE','LOC'])&('LOC' in entity):
            tokens.append((token.text,'Location','#fda'))

        elif (token.ent_type_ == 'ORG') & ('ORG' in entity):
            tokens.append((token.text,'Organisation','#afa'))

        elif (token.ent_type_ == 'NUMBER') & ('NUM' in entity):
            tokens.append((token.text,'Number','#aaf'))

        else:
            tokens.append(" "+ token.text + ' ')

    if anonymize:
        anonymized_text = []

        for text in tokens:
            if type(text) == tuple:
                anonymized_text.append(('D'*len(text[0]),text[1],text[2]))
            else:
                anonymized_text.append(text)
        
        return anonymized_text

    return tokens



text = st.text_area('Type something here to anonymize OR'," ",5,max_chars=1000,)
txt_path = st.file_uploader('Upload a file to anonymize',["docx",'txt','pdf','doc'],False)

if txt_path is not None:
    text = txt_path.getvalue()
    text = text.decode('utf-8')

st.write(text)

languages = st.sidebar.selectbox('Select Language',['English','French','Spanish','Hindi'])

entities = st.sidebar.multiselect('Select a set of Entities',['LOC','PER','ORG','NUM'],['LOC','PER','ORG'])

model = lang_models()
selected_mod = model[languages]

doc = selected_mod(text)

result = process_data(doc,entities)
result
# annotated_text(*result)

anonymize = st.sidebar.checkbox('Anonymize')
if anonymize:
    st.markdown('**Anonymized Text**')
    st.markdown('-----')
    anonymize_result = process_data(doc,entities,True)
    anonymize_result
    # annotated_text(*anonymize_result)