import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit 
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Normalizer, FunctionTransformer, StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
 
#*********** Customer Preprocessing ********
def genre_to_int(data):  
    genre = data['genre']
    if genre == 'Pop': return 0
    if genre == 'Hip-Hop': return 1
    if genre == 'Rock': return 2
    if genre == 'Metal': return 3
    if genre == 'Country': return 4
    if genre == 'Jazz': return 5
    if genre == 'Electronic': return 6
    if genre == 'Folk': return 7
    if genre == 'R&B': return 8
    if genre == 'Indie': return 9
    if genre == 'Not Available': return None 
    if genre == 'Other': return None
    

def get_stemmed(sentence): 
    from nltk.stem.porter import PorterStemmer
    tokens = sentence.split() 
    stemmer = PorterStemmer() 
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return " ".join(stemmed_tokens)

def get_lemma(sentence): 
    from nltk.stem import WordNetLemmatizer
    tokens = sentence.split() 
    lemma = WordNetLemmatizer() 
    lemmad_tokens = [lemma.lemmatize(token) for token in tokens]
    return " ".join(lemmad_tokens)
    
#*************** Read Data *******************
csv = pd.read_csv('./final_final_lyrics_dataframe.csv')

# csv['genre'] = csv.apply(genre_to_int, axis=1)
print(len(csv))
csv.dropna(axis = 0, how="any", inplace=True) #drop row if any null in it 
print(len(csv))
labels = csv['genre']
lyrics = csv['lyrics']

# #**************** Preprocessing **************** 

# def stem_text(stem):  
#     lyrics['lyrics'] = lyrics['lyrics'].apply(get_stemmed, axis=1)

# #**************** Split Data ******************* 
X_train, X_test, y_train, y_test = train_test_split(lyrics, labels, test_size = .3, shuffle=True, random_state = 43)

# #**************** Models ********************** 
clf = LogisticRegression(multi_class='auto')


# #*************** Validation Pipeline *************
pipeline = Pipeline([
    ('features_union', FeatureUnion([
                ('ngrams_feature', Pipeline([('ngrams_vect', TfidfVectorizer(binary = True, ngram_range=(1,1)))
            # ])),
            #     ('stemming', Pipeline([('stem', FunctionTransformer(get_stemmed, validate=False))
            # ])), 
            #     ('lemmatize', Pipeline([('lemma', FunctionTransformer(get_lemma, validate=False))        
    ]))])),
    #('bow', CountVectorizer(lowercase=True, stop_words='english')),
    #('normalization', Normalizer(copy=False)), 
    ('classifier', clf)
])

parameters_grid = {'classifier__penalty': ('l2', 'l1') 
                    } #'features_union__tfidf__ngram_range': ((1,1), (1,2))

print('Starting to train grid search')
grid_search = GridSearchCV(pipeline, parameters_grid, cv = 2, n_jobs = 1, scoring="accuracy")
grid_search.fit(X_train, y_train)

print("**********************TRAINING CLASSIFICATION REPORT *********************")
print("model: ", grid_search.best_estimator_)
print("training score: ", grid_search.best_score_ , '\n', "best parameters: ", grid_search.best_params_,'\n')
print("**********************VALIDATION REPORT *********************")
cvres = grid_search.cv_results_
for accuracy, params in zip(cvres['mean_test_score'],cvres['params']):
    print('Mean accuracy: ', accuracy,'  using: ',params)

best_model = grid_search.best_estimator_
y_true, y_pred = y_test, best_model.predict(X_test)
print('*******************TEST CLASSIFICATION REPORT****************')
print(classification_report(y_true, y_pred))

titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]

class_names = [0,1,2,3,4,5,6,7,8,9]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(grid_search, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()