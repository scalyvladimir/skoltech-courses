from ast import literal_eval
import re
import pandas as pd
import nltk

stopwords = nltk.corpus.stopwords.words('english')

#removing stop words 
def remove_stopwords(text):
    output= [i.lower() for i in text.split() if i not in stopwords]
    return " ".join(output)

#moving some integer variants into str
def binning(col, cut_points, labels=None):
    #Define min and max values:
    minval = col.min()
    maxval = col.max()
    #create list by adding min and max to cut_points
    break_points = [minval] + cut_points + [maxval]
    #if no labels provided, use default labels 0 ... (n-1)
    if not labels:
        labels = range(len(cut_points)+1)
    #Binning using cut function of pandas
    colBin = pd.cut(col, bins=break_points, labels=labels, include_lowest=True)
    return colBin


def load_profiles_reviews(reviews_path, profiles_path):
    profiles = pd.read_csv(profiles_path)
    reviews = pd.read_csv(reviews_path)
    
    # create mappinng for assigning numeric id to profile_name
    profile_cat = profiles['profile'].astype('category').cat

    profiles = (
        profiles
        .drop_duplicates(subset=['profile'], keep='last')
        .assign(profile_idx = profile_cat.codes) # map index to profile names
        # We have a column with array of the favourite animes. each element of this array is anime_id
        # which is to be integer but it is stored in string format. We fix that
        .assign(favorites_anime = lambda x: (
            x["favorites_anime"].apply(
                    lambda favorites_anime_line: [int(x) for x in literal_eval(favorites_anime_line)]
                )
            )
        )
    )    
    
    reviews = (
        reviews
        # dropping duplicates (we have plenty of them)
        .drop_duplicates(subset=['profile', 'anime_uid', 'text', 'score'])
        .rename(columns = {'anime_uid':'anime_id'})
        .assign( # map index to profile names
            profile_idx = lambda x: profile_cat.categories.get_indexer(x['profile'])
        )
    )

    assert reviews["profile_idx"].min() >= 0 # for invalid profile entries `get_indexer` assings -1
    return reviews, profiles


def load_animes(path):
    
    #We gather all the text information into one column
    def create_soup(x):
        return str(x['synopsis_ok']) + "\n" + str(x['genre'])  + "\n" + str(x['title']) 

    anime = (
        pd.read_csv(path)
        #renaming some columns     
        .rename(columns = {'uid':'anime_id' })
        #dropping duplicates and unnecessary columns
        .drop_duplicates(subset=['title', 'synopsis','link','score'])
        .drop(columns=['link', 'img_url','aired', 'episodes','popularity'])
        # create auxiliary fields
        .assign(
            # construct a single line of genres (comma-separated)
            genre = lambda x: x['genre'].apply(lambda genre_line: ", ".join([x.strip() for x in literal_eval(genre_line)])),
            # remove stopwords from sinopsis
            synopsis_ok = lambda x: x['synopsis'].map(str).apply(lambda x: remove_stopwords(x)),
            # creating soup and preprocossing the info
            soup = lambda x: (
                x
                .apply(create_soup, axis =1)
                .apply(lambda x : re.sub('[^A-Za-z]+', ' ', str(x)))
                .map(str).apply(lambda x: " ".join(x for x in x.split() if x not in stopwords))
            )
        )
    )
    return anime