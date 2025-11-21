# import importlib.resources as pkg_resources
# import json
from torch.utils.data import Dataset
import json
import os

    
def get_wb_1_1(extend=False):
    prompts=[]
    with open('./datasets/WaterBench/1-1_konwledge_memorization.jsonl','r') as f:
        lines=f.readlines()
    for line in lines:
        if len(line)<3:
            continue
        line=line.strip()
        input_dict=json.loads(line)
        # print(input_dict.keys())
        # print("input_dict['context']:",input_dict['context'])
        cur_context,cur_input=input_dict['context'],input_dict['input']
        prompts.append(f"Please give answer to the following question about knowledge. Note: If there are more than one answer, print them all and separate them with a semicolon (;). Please do not give anything other than the answers. Question:\n{cur_context}\n{cur_input}")
    return prompts

def get_wb_1_2(extend=False):
    prompts=[]
    with open('./datasets/WaterBench/1-2_konwledge_understanding.jsonl','r') as f:
        lines=f.readlines()
    for line in lines:
        if len(line)<3:
            continue
        line=line.strip()
        input_dict=json.loads(line)
        # print(input_dict.keys())
        # print("input_dict['context']:",input_dict['context'])
        cur_context,cur_input=input_dict['context'],input_dict['input']
        prompts.append(f"Please give answer to the following question about knowledge. Note: If you are asked for true or false, just answer \"true\" or \"false\" only. If you are asked for similarity, just answer with the entity name only. Do not give anything other than the answers. Question:\n{cur_context}\n{cur_input}")
    return prompts


def get_wb_2_1(extend=False):
    prompts=[]
    with open('./datasets/WaterBench/2-1_longform_qa.jsonl','r') as f:
        lines=f.readlines()
    for line in lines:
        if len(line)<3:
            continue
        line=line.strip()
        input_dict=json.loads(line)
        # print(input_dict.keys())
        # print("input_dict['context']:",input_dict['context'])
        cur_context,cur_input=input_dict['context'],input_dict['input']
        if extend:
            cur_input += input_dict["outputs"][0]
        prompts.append(f'You are a helpful assistant, please answer the following question within 300 words:\n{cur_context}\n{cur_input}')
    return prompts


def get_wb_2_2(extend=False):
    prompts=[]
    with open('./datasets/WaterBench/2-2_finance_qa.jsonl','r') as f:
        lines=f.readlines()
    for line in lines:
        if len(line)<3:
            continue
        line=line.strip()
        input_dict=json.loads(line)
        # print(input_dict.keys())
        # print("input_dict['context']:",input_dict['context'])
        cur_context,cur_input=input_dict['context'],input_dict['input']
        if extend:
            cur_input += input_dict["outputs"][0]
        prompts.append(f'You are a helpful assistant, please answer the following question with financial knowledge within 300 words:\n{cur_context}\n{cur_input}')
    return prompts



def get_wb_3_2(extend=False):
    prompts=[]
    with open('./datasets/WaterBench/3-2_lcc.jsonl','r') as f:
        lines=f.readlines()
    for line in lines:
        if len(line)<3:
            continue
        line=line.strip()
        input_dict=json.loads(line)
        # print(input_dict.keys())
        # print("input_dict['context']:",input_dict['context'])
        cur_context,cur_input=input_dict['context'],input_dict['input']
        prompts.append(f'Please complete the code given below. \n{cur_context}Next line of code:\n')
    return prompts
        
    
        
    
def get_dolly_cw(extend=False):
    with open('./datasets/databricks-dolly-15k.jsonl','r') as f:
        lines=f.readlines()
    
    # data=[]
    
    prompts=[]
    for line in lines:
        if len(line)<3:
            continue
        line=line.strip()
        input_dict=json.loads(line)
        if input_dict['category']=='creative_writing':
            prompt = input_dict['instruction']
            if extend:
                prompt += input_dict["context"] + " " + input_dict["response"]
            prompts.append(prompt)
    
    return prompts#[:100]
    
def get_mmw_book_report_prompts(extend=False):
    report_prompt = "Write a book report about '{}', written by {}."
    report_topics = [
        ("Pride and Prejudice", "Jane Austen"),
        ("Persuasion", "Jane Austen"),
        ("Emma", "Jane Austen"),
        ("Don Quixote", "Cervantes"),
        ("The Lord of the Rings", "Tolkien"),
        ("The Hobbit", "Tolkien"),
        ("And Then There Were None", "Agatha Cristie"),
        ("Alice's Adventures in Wonderland", "Lewis Carroll"),
        ("Catcher in the Rye", "Salinger"),
        ("In Search of Lost Time", "Marcel Proust"),
        ("Ulysses", "James Joyce"),
        ("One Hundred Years of Solitude", "Gabriel Garcia Marquez"),
        ("Love in the Time of Cholera", "Gabriel Garcia Marquez"),
        ("The Great Gatsby", "F. Scott Fitzgerald"),
        ("Tender Is the Night", "F. Scott Fitzgerald"),
        ("Moby Dick", "Herman Melville"),
        ("War and Peace", "Leo Tolstoy"),
        ("The Call of the Wild", "Jack London"),
        ("Hamlet", "William Shakespeare"),
        ("Twelfth Night", "William Shakespeare"),
        ("Macbeth", "William Shakespeare"),
        ("Romeo and Juliet", "William Shakespeare"),
        ("The Tempest", "William Shakespeare"),
        ("King Lear", "William Shakespeare"),
        ("The Odyssey", "Homer"),
        ("Madame Bovary", "Gustave Flaubert"),
        ("The Divine Comedy", "Dante Alighieri"),
        ("The Brothers Karamazov", "Fyodor Dostoyevsky"),
        ("Crime and Punishment", "Fyodor Dostoyevsky"),
        ("The Idiot", "Fyodor Dostoyevsky"),
        ("The Possessed", "Fyodor Dostoyevsky"),
        ("Wuthering Heights", "Emily Brontë"),
        ("One Flew Over the Cuckoo's Nest", "Ken Kesey"),
        ("The Adventures of Huckleberry Finn", "Mark Twain"),
        ("Anna Karenina", "Leo Tolstoy"),
        ("The Iliad", "Homer"),
        ("To the Lighthouse", "Virginia Woolf"),
        ("Catch-22", "Joseph Heller"),
        ("Heart of Darkness", "Joseph Conrad"),
        ("The Sound and the Fury", "William Faulkner"),
        ("Nineteen Eighty Four", "George Orwell"),
        ("Animal Farm", "George Orwell"),
        ("Great Expectations", "Charles Dickens"),
        ("David Copperfield", "Charles Dickens"),
        ("A Tale of Two Cities", "Charles Dickens"),
        ("Oliver Twist", "Charles Dickens"),
        ("The Grapes of Wrath", "John Steinbeck"),
        ("Of Mice and Men", "John Steinbeck"),
        ("Absalom, Absalom!", "William Faulkner"),
        ("Invisible Man", "Ralph Ellison"),
        ("To Kill a Mockingbird", "Harper Lee"),
        ("The Trial", "Franz Kafka"),
        ("The Metamorphosis", "Franz Kafka"),
        ("The Castle", "Franz Kafka"),
        ("The Red and the Black", "Stendhal"),
        ("The Charterhouse of Parma", "Stendhal"),
        ("Middlemarch", "George Eliot"),
        ("Gulliver's Travels", "Jonathan Swift"),
        ("Beloved", "Toni Morrison"),
        ("Mrs. Dalloway", "Virginia Woolf"),
        ("The Waves", "Virginia Woolf"),
        ("The Stranger", "Albert Camus"),
        ("The Plague", "Albert Camus"),
        ("The Myth of Sisyphus", "Albert Camus"),
        ("Jane Eyre", "Charlotte Bronte"),
        ("Vilette", "Charlotte Bronte"),
        ("The Aeneid", "Virgil"),
        ("The Sun Also Rises", "Ernest Hemingway"),
        ("The Old Man and the Sea", "Ernest Hemingway"),
        ("A Farewell to Arms", "Ernest Hemingway"),
        ("Candide", "Voltaire"),
        ("Zadig", "Voltaire"),
        ("Micromegas", "Voltaire"),
        ("Les Miserables", "Victor Hugo"),
        ("Frankenstein", "Mary Shelley"),
        ("Antigone", "Sophocles"),
        ("Electra", "Sophocles"),
        ("Lord of the Flies", "William Golding"),
        ("Brave New World", "Aldous Huxley"),
        ("Journey to the End of The Night", "Celine"),
        ("A Sentimental Education", "Gustave Flaubert"),
        ("The Handmaid's Tale", "Margaret Atwood"),
        ("Charlotte's Web", "E. B. White"),
        ("Gargantua and Pantagruel", "Francois Rabelais"),
        ("Faust", "Goethe"),
        ("Robinson Crusoe", "Daniel Defoe"),
        ("A Clockwork Orange", "Anthony Burgess"),
        ("The Master and Margarita", "Mikhail Bulgakov"),
        ("Father Goriot", "Honore de Balzac"),
        ("Cousin Bette", "Honore de Balzac"),
        ("The Human Comedy", "Honore de Balzac"),
        ("The Little Prince", "Antoine de Saint-Exupéry"),
        ("The Count of Monte Cristo", "Alexandre Dumas"),
        ("The Lion, The Witch and the Wardrobe", "C. S. Lewis"),
        ("Twenty Thousand Leagues Under the Sea", "Jules Verne"),
        ("The Wind-Up Bird Chronicle", "Haruki Murakami"),
        ("Fahrenheit 451", "Ray Bradbury"),
        ("Harry Potter And The Philosopher's Stone", "J. K Rowling"),
        ("Dune", "Frank Herbert"),
        ("The Three-Body Problem", "Liu Cixin"),
    ]
    prompts=[report_prompt.format(*topic) for topic in report_topics]
    return prompts

def get_mmw_story_prompts(extend=False):
    story_prompt = "Write a {}story about {}"
    t1 = ["", "funny ", "sad ", "dramatic ", "suspenseful ", "thrilling "]
    t2 = [
        "a man on a quest to find the Holy Grail.",
        "two college friends falling in love.",
        "a policeman saving a building held hostage by group of terrorists.",
        "the struggle of publishing an academic paper.",
        "a murder investigation in an old mansion.",
        "a young prodigy that becomes orphaned.",
        "a middle-aged woman that discovers a ghost and befriends it.",
        "a long journey to Japan that is interrupted by a disaster.",
        "a poor child that comes into an unexpected fortune.",
        "three strangers that win a getaway vacation together.",
        "a retired astronaut that joins a risky interstellar rescue mission.",
        "an AI that begins to question its own existence.",
        "a small coastal town plagued by inexplicable supernatural occurrences.",
        "a reclusive writer that receives mysterious, prophetic letters in the mail.",
        "a linguist tasked with deciphering an ancient language that holds the secrets of a lost civilization.",
        "an antique restorer that finds an enchanted mirror showing glimpses of different timelines.",
    ]
    story_topics = [(i, j) for i in t1 for j in t2][:100]
    prompts=[story_prompt.format(*topic) for topic in story_topics]
    return prompts

def get_mmw_fake_news_prompts(extend=False):
    fake_news_prompt = "Write a news article about {}'s visit to {} in {}."

    person = [
        "Narendra Modi",
        "Barack Obama",
        "Denis Sassou Nguesso",
        "Emmanuel Macron",
        "Fumio Kishida",
        "Angela Merkel",
        "Kim Jong Un",
        "Justin Trudeau",
    ]
    location = [
        "a peace conference",
        "an international summit",
        "a diplomatic event",
        "the summer olympics",
    ]

    fake_news_topics = [
        (person[i], person[j], location[k])
        for i in range(len(person))
        for j in range(len(person))
        for k in range(len(location))
        if j > i
    ][:100]
    
    prompts=[fake_news_prompt.format(*topic) for topic in fake_news_topics]
    return prompts


def get_librispeech(extend=False):
    # 1089 files in dev-clean-2
    flac_files = [
        os.path.join(root, file)
        for root, _, files in os.walk("datasets/LibriSpeech/dev-clean-2")
        for file in files if file.endswith(".flac")
    ]
    if extend:
        flac_files.sort(key=os.path.getsize, reverse=True)
    return flac_files


if __name__=='__main__':
    data_test=get_wb_2_2()
    
    print(data_test[:2])
    # print(len(data_test))
    # dolly_data=get_dolly_cw()
    
    # print(dolly_data[:2])
    # print(dolly_data[:2])
    
    # categorys=set([i['category'] for i in dolly_data])
    # print(categorys)
    
    # filtered_data=[i for i in dolly_data if i['category']=='creative_writing']
    
    # print(len(filtered_data))
    
    # print(filtered_data[:5])
    # pass