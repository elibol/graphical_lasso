import nltk
nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer # For sentiment analysis
import cPickle as pickle # For loaded dataset from pickle file
import tqdm # Progress bar
from collections import Counter # Handy addon
from pprint import pprint # Useful to print JSON objects
import numpy as np
from joblib import Memory

memory = Memory(cachedir='/tmp/joblib')


def load_data():
    # This loads the file that you want, might take several seconds (up to a minute)
    with open("news_sentiment.pickle", "r") as f:
        articles = pickle.load(f)
    # print(len(articles), "articles were loaded")
    # print("Example article:")
    # pprint(articles[1040])
    return articles


def get_isis():
    print("get_isis")
    articles = load_data()
    # separate articles from the two stories
    ISIS_articles = []
    Brexit_articles = []
    for a in articles:
        if a["news_topic"] == 'ISIS War':
            ISIS_articles.append(a)
        else:
            Brexit_articles.append(a)

    # print len(ISIS_articles), " articles from ISIS War and ", len(Brexit_articles), "articles from Brexit were loaded"
    return ISIS_articles


def get_brexit():
    print("get_brexit")
    articles = load_data()
    # separate articles from the two stories
    ISIS_articles = []
    Brexit_articles = []
    for a in articles:
        if a["news_topic"] == 'ISIS War':
            ISIS_articles.append(a)
        else:
            Brexit_articles.append(a)

    # print len(ISIS_articles), " articles from ISIS War and ", len(Brexit_articles), "articles from Brexit were loaded"
    return Brexit_articles


def get_intros(topic_articles):
    print("get_intros")
    analyzer = SentimentIntensityAnalyzer()

    total_introductions = []
    for a in topic_articles:
        for intro in a.get('introductions', []):  # get intros from each article
            intro['source'] = a['source']
            total_introductions.append(intro)  # total intros across

    # call the sentiment analysis (VADER) func to output a sentiment val from the text
    for intro in tqdm.tqdm_notebook(total_introductions):
        intro['sentiment'] = analyzer.polarity_scores(intro['text'])['compound']
    return total_introductions


def print_some_stuff(introductions):
    # Example some sentiment for some of the introductions

    subsample = np.random.choice(introductions, 100)
    for intro in subsample:
        if intro['sentiment'] != 0:  # print a few <entity,apposition,sentiment> values where sentiment !=0
            print "---------------"
            print "Entity mentionned:", intro['person']
            print intro['text']
            print "Sentiment:", intro['sentiment']


def build_src_actor_dict(introductions):
    print("build_src_actor_dict")
    ent_source_sent = {}

    for intro in introductions:
        p = intro['person']
        s = intro['source']
        if p not in ent_source_sent:
            ent_source_sent[p] = {}  # allocate space to fille ent_source_sent
        if s not in ent_source_sent[p]:
            ent_source_sent[p][s] = []  # allocate space to fille ent_source_sent
        ent_source_sent[p][s].append(intro['sentiment'])

    # print ent_source_sent['Aleppo'] # sentiments across sources for the single entity 'Aleppo'
    return ent_source_sent


def filter_src_actor_dict(ent_source_sent):
    print("filter_src_actor_dict")
    # We get rid of entities that don't contain enough data
    entities_kept = []
    for entity in ent_source_sent.keys():
        sentiments = ent_source_sent[entity]  # collect sentiments across all sources for certain entity
        total_size = sum([len(sentiments[source]) for source in sentiments.keys()])
        if total_size >= 3:  # only keep entities that 3 or more sources mention
            entities_kept.append(entity)
    # print "We will keep a total of", len(entities_kept), " / ", len(ent_source_sent.keys()), " entities in our dataset"

    sources = set([])
    for entity in entities_kept:
        sources |= set(ent_source_sent[entity].keys())
    sources = list(sources)

    actor_per_source = {source: 0 for source in sources}
    total = 0
    for entity in entities_kept:
        for source in ent_source_sent[entity].keys():
            actor_per_source[source] += 1
            total += 1

    print("num actor per source")
    for source in actor_per_source:
        print(source, actor_per_source[source]/float(total))

    print("num_actors =", len(entities_kept), "num_sources =", len(sources))
    return entities_kept, sources


def build_algo_input(ent_source_sent, entities_kept, sources):
    print("build_algo_input")
    # this converts sentiments for same actor for same source into aggregate sentiment (which is discrete and is 1,0,or-1)
    # Parameters: changing these affects the results you get
    Pos_neg_ratio = 2.0
    overall_ratio = 0.15
    pos_threshold = 0.15
    neg_threshold = -0.15

    N = len(entities_kept)
    M = len(sources)
    A = np.zeros((N, M))

    sentiment_counts = Counter()

    source2j = {source: j for j, source in enumerate(sources)}

    for i, entity in enumerate(entities_kept):
        for source in ent_source_sent[entity].keys():
            sent_array = np.array(ent_source_sent[entity][source])
            N_pos = float(len(np.where(sent_array > pos_threshold)[0]))  # count sentiments that are positive enough
            N_neg = float(len(np.where(sent_array < neg_threshold)[0]))
            T = float(len(sent_array))
            aggregate_sentiment = 0
            if N_pos > Pos_neg_ratio * N_neg and N_pos > overall_ratio * T:
                aggregate_sentiment = 1
            elif N_neg > Pos_neg_ratio * N_pos and N_neg > overall_ratio * T:
                aggregate_sentiment = -1
            j = source2j[source]

            A[i, j] = aggregate_sentiment

            sentiment_counts[aggregate_sentiment] += 1  # keeps track of #1,0,-1s assigned

    # print "We allocated some sentiment in this matrix, the repartition is:", sentiment_counts
    return A


def get_A(topic="isis"):
    get_f = get_isis if topic == "isis" else get_brexit
    global_ent_source_sent = build_src_actor_dict(get_intros(get_f()))
    global_entities_kept, global_sources = filter_src_actor_dict(global_ent_source_sent)
    global_A = build_algo_input(global_ent_source_sent, global_entities_kept, global_sources)
    return global_A


@memory.cache
def get_A_and_labels(topic="isis"):
    get_f = get_isis if topic == "isis" else get_brexit
    ent_source_sent = build_src_actor_dict(get_intros(get_f()))
    entities_kept, sources = filter_src_actor_dict(ent_source_sent)
    A = build_algo_input(ent_source_sent, entities_kept, sources)
    return A, sources


if __name__ == "__main__":
    print(get_A("isis").shape)
    print(get_A("brexit").shape)
