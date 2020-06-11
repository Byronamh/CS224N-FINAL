# CS224N-FINAL
Final project for CS224N. The project extracts data from resumes using en [ELMo](https://allennlp.org/elmo)


# Custom ELMo NER model for job matching
A tool for extracting data from resumes and finding the best job opportunity

----------

### Byron Andrés Mota Hernández - 15246

## Abstract
The inner workings of job search engines like [Linkedin](Linkedin.com) and [Indeed.com](https://www.indeed.com/) have always peaked my interest. How do they find jobs that are suited for the user? What skills makes a user ideal for a specific job? In this paper, NLP, CNN and NER are used to extract data from resumes, train a model and finally match the best potential jobs, giving an insight on how enterprise solutions might be built.

## Introduction
Nowadays, there are many job search engines that require minimal user data, just an updated resume. These search engines have methods to extract work, education and general expertise from  uploaded resumes. Additionally, whenever a user searches for a job without a description or title (using Linkedin), it will automatically find potential job interests in the defined área (if one isn't specified, it does a world wide search). 

The original scope of this project was to match job requests with applicants and applicants with job requests. However, the scope would grow too large, as a training and testing dataset was needed for both aplicants and job posters, so it was reduced to finding relationships between job titles found within resumes.

Developing a funcional system similar in any way to Linkedin would be impossible in the given time, but developing a core module that could be used and extended into a fully functional system is far more feasable.

## Core modules used in this project
- [pandas](https://pandas.pydata.org/)            - for csv loading and reading
- [numpy](https://numpy.org/)             - for efficient array and tuple manipulation
- [tensorflow_hub](https://www.tensorflow.org/hub)    - as RNN main engine
- [tensorflow](https://www.tensorflow.org/)        - as ML framework backend
- [keras](https://keras.io/)             - as ML framework
- [tika](https://pypi.org/project/tika/)              - for extraction of data from pdf files

## Related Work
NER is a field of NLP that's in constant development, as there's a lack of formally described techniques for identifying named entities that appear within natural language sentences.
Multiple libraries have been developed, such as [SpaCy](https://spacy.io/) and [OpenNLP](https://opennlp.apache.org/), however they're still young, having less than 10 years of age as of 2020.
An approach to [GloVe](https://nlp.stanford.edu/projects/glove/) (global vectos for word representation) called ELMo that simplifies NER has been in discussion for [quite a while now](https://arxiv.org/pdf/1802.05365.pdf).

This project was heavily inspired by [this tutorial](https://www.analyticsvidhya.com/blog/2019/03/learn-to-use-elmo-to-extract-features-from-text/)

## Approach
[Allen NLP](https://allennlp.org/elmo) describes ELMo as:

> <em> "ELMo is a deep contextualized word representation that models both (1) complex characteristics of word use (e.g., syntax and semantics), and (2) how these uses vary across linguistic contexts (i.e., to model polysemy). These word vectors are learned functions of the internal states of a deep bidirectional language model (biLM), which is pre-trained on a large text corpus. They can be easily added to existing models and significantly improve the state of the art across a broad range of challenging NLP problems, including question answering, textual entailment and sentiment analysis." [1](https://allennlp.org/elmo)</em>

Elmo is a method that looks at the whole context (sentnce) instead of fixed embedding for each word in the contex, this is called Language Modeling. I'ts a bi-directional LSTM trained specifically for these embeddings without the need of tokenized words or labels. 

![Elmo overview](https://github.com/Byronamh/CS224N-FINAL/blob/master/images/elmo.png?raw=true)

>Overview of how ELMo diagrams are structured

![Elmo inner diagram](https://github.com/Byronamh/CS224N-FINAL/blob/master/images/elmoDiagram.png?raw=true)

>ELMo trains a bi-directional LSTM – that way, it's language model has context of the previous and next word.

> Inspiration for both diagrams was taken from [Jay Alammar](http://jalammar.github.io/illustrated-bert/)

## Experiments

### Data
For the initial ELMo training will be done with the [Kaggle NER dataset](https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus/home).
After the NER dataset has been loaded, it will be fed 220 unique resumes that were already processed by a trained spaCy instance with the following custom labels:
 - `Name`
 - `College Name`
 - `Degree`
 - `Graduation Year`
 - `Years of Experience`
 - `Companies worked at`
 - `Designation`
 - `Skills`
 - `Location`
 - `Email Address`
 
 This human labeled dataset was can be found on [dataturks](https://dataturks.com/projects/abhishek.narayanan/Entity%20Recognition%20in%20Resumes)

### Evaluation Method
A comparison with the originally trained model and the custom label trained model will be done. Favoring the usage of the StanfordNLP labels over the custom ones.

### Results

`python elmoBaseTrainer.py`

The training of the initial ELMo with the english corpus yielded an initial accuracy of 0.9304 during the first epoch of training.
After 5 epochs of training, the model accuracy increased to 0.9572
```
Train on 19408 samples, validate on 2160 samples
Epoch 1/5
19408/19408 [==============================] - loss: 0.1419 - acc: 0.9304 - val_loss: 0.0630 - val_acc: 0.9404
Epoch 2/5
19408/19408 [==============================] - loss: 0.0552 - acc: 0.9434 - val_loss: 0.0513 - val_acc: 0.9499
Epoch 3/5
19408/19408 [==============================] - loss: 0.0462 - acc: 0.9491 - val_loss: 0.0480 - val_acc: 0.9521
Epoch 4/5
19408/19408 [==============================] - loss: 0.0417 - acc: 0.9593 - val_loss: 0.0462 - val_acc: 0.9570
Epoch 5/5
19408/19408 [==============================] - loss: 0.0388 - acc: 0.9572 - val_loss: 0.0446 - val_acc: 0.9623
```

> Output of running initial ELMo vocab training.

Once the model was trained for the english language, It was time to Load the resume dataset along with the new labels/tags

The structure of the generated model is the following:

`model.summary()`

```
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            [(None, 50)]         0
__________________________________________________________________________________________________
lambda (Lambda)                 (16, None, 1024)     0           input_1[0][0]
__________________________________________________________________________________________________
bidirectional (Bidirectional)   (16, None, 1024)     6295552     lambda[0][0]
__________________________________________________________________________________________________
bidirectional_1 (Bidirectional) (16, None, 1024)     6295552     bidirectional[0][0]
__________________________________________________________________________________________________
add (Add)                       (16, None, 1024)     0           bidirectional[0][0]
                                                                 bidirectional_1[0][0]
__________________________________________________________________________________________________
time_distributed (TimeDistribut (None, None, 768960) 788184000   add[0][0]
==================================================================================================
Total params: 800,775,104
Trainable params: 800,775,104
Non-trainable params: 0
__________________________________________________________________________________________________
```
---

Model Visualization using keras.utils.plot_model

![Model Visualization](https://github.com/Byronamh/CS224N-FINAL/blob/master/images/model_plot.png?raw=true)


`python elmoResumeTrainer.py`

After loading the trained model and processing all the words in the corpus, prediction of the new set are made:
The new dataset introduces the following new labels:
 - `Name`
 - `College Name`
 - `Degree`
 - `Graduation Year`
 - `Years of Experience`
 - `Companies worked at`
 - `Designation`
 - `Skills`
 - `Location`
 - `Email Address`
 
This is a comparison of the model prediction of the same text before and after adding the custom labels
```
Word            Pred : (True)
==============================
Abhishek        B-per (B-per)
Jha             B-per (B-per)
Application     O     (O)
Development     B-geo (B-geo)
Associate       O     (O)
-               O     (B-geo)
Accenture       O     (O)
Bengaluru       O     (O)
,               O     (O)
Karnataka       B-geo (B-org)
```
> original ELMo

```
Abhishek	Name
Jha	        Name
Application	Designation
Development	Designation
Associate	Designation
-	        O
Accenture	Companies worked at
Bengaluru	Location
,	        O
Karnataka	O
```
> input sample training data

```
Word            Pred 
==============================
Abhishek	    B-per
Jha	            B-per
Application	    Designation
Development	    Designation
Associate	    Designation
-	            O
Accenture	    Companies worked at
Bengaluru	    Location
,	            O
Karnataka	    B-geo
```
> ELMo with custom label training

### Analysis

As you can see, some predicted tags changed, mostly those that used to be simple non named objects.
There's conflict in context definition with `Location` and `B-geo`, as they mean the same thing, but the name of the tags are different (this also happens with `B-org` - `Companies worked at`, `gpe` - `Location` and other labels).

This model did extremely well with a high accuracy due to the implementation of the [tensorflow hub ELMo](https://tfhub.dev/google/elmo/3) as the core model

The model did some mix and matching between new and old tags, proving the flexibility of this NER architecture.

### Conclusions for future work
- A mapping between the custom tags and the StanfordNRE standard tags should be made, to avoid tag overriding (as seen above with `Name` and `per`). This could be achieved by implementing a simple mapper that matches ocurrances of custom tags to the standarized ones.
- The names of any custom tags implemented into the model should be shortened (`Designation` to `des`, `Graduation year` to `gdy` and so on).
-  ELMo has proven to be capable enough to understand complex relationships between words, for example 'Devops Engineer' forming a same `Designation` in the given context yet modify it's existing tag relations with newly implemented ones. This type of flexibility and lack of language tokenization opens the door for context processing in other languages with vastly different grammatical structures (Spanish, Chinese, Welsh, etc.).
- This project mostly uses tensorflow v1, as tue current LTS is v2, some design features should be modified to adapt to new standards
