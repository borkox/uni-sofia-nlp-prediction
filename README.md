
#NLP : Word prediction

## Intro
This is a NLP university software project
 to be presented in Sofia University.
Author: Borislav Markov, Bulgaria, Sofia.
 
It is not designed to run in production mode, it is just 
for demonstration.

TASK: **We need to generate next word by given sequence of words.**

For the task I use a pre-selected corpus.
I also put the pre-trained network, so one can 
start by running examples directly.

## Implementation
### Tech stack and prerequisites
 * Java 11 - must be installed
 * Maven 3 - must be installed
 * Dependent libraries - will be downloaded on build time
 
### Build
```shell script
mvn clean install
```

### Run predictions
Running directly predictions will run on saved network.
```shell script
mvn exec:java -Dexec.mainClass="bg.unisofia.bsmarkov.nlp.RunPrediction"
```

### Other commands
Rebuilding lookup table and scaled word vectors:
```shell script
mvn exec:java -Dexec.mainClass="bg.unisofia.bsmarkov.nlp.MakeLookupTable"
```
Training the network on train dataset(will require 1-2h on Intel 9):
```shell script
mvn exec:java -Dexec.mainClass="bg.unisofia.bsmarkov.nlp.TrainNet"
```
Evaluation of the network on the two datasets is started like this:
```shell script
mvn exec:java -Dexec.mainClass="bg.unisofia.bsmarkov.nlp.EvalNet"
```

### Deep learning framework
I used Java library DL4J, https://deeplearning4j.konduit.ai/

Implementation involves an artificial neural network with LSTM layers.

### Corpus
The corpus of sentences for the learning I took from DL4J examples.
The original corpus is NLPDATA, but I took only one file from it `raw_sentences.txt`
and then I made 2 files from it:
* raw_sentences_train.txt - first 500 lines to train my network.
* raw_sentences_test.txt - another 100 lines to train my network (unseen data).
I still use the big file to make lookup table of word vectors.
Corpus is saved in `src/main/resources`.

Corpus source:
https://github.com/eclipse/deeplearning4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/utils/DownloaderUtility.java#L57

## Algorithm
First I build a lookup table of words by parsing the corpus.
Tokenization of raw sentences is done with just witespace. 
Lowercase filter is used. Punctuation is not removed.
Then I train SkipGram algorithm to give me word representation
as vectors of size 150. After that I scale the vector
representations to be between [0,1].
Lookup table is stored in a binary file 'persist/lookupTable'.

The main recurrent neural network (RNN) is constructed in the train phase.
Its purpose is to be trained on a supervised learning by giving 
as an input a word, and on the output to have next word in the sequence.
This is a supervised learning through time(through sequence).
 
Network In/out:
* Input : a word vector from 'lookup table' of size 150.
* Output: the next word as a one-hot vector of size 250 (depending of size of lookup table)

One can ask: **why then the input is not a one-hot vector as the output?**
The answer is simple. Since I use a fixed lookup table made with SkipGram
which means that **similar words are represented with similar vectors**, so
the RNN would guess right even unseen data correctly. If I just used 
one-hot vectors for the input, then RNN would return just learnt data 
and nothing more.

### Neural Network Architecture:
<pre>

======================================================================================
LayerName (LayerType)     nIn,nOut   TotalParams   ParamsShape                        
======================================================================================
layer0 (LSTM)             150,80     73 920        W:{150,320}, RW:{80,320}, b:{1,320}
layer1 (LSTM)             80,80      51 520        W:{80,320}, RW:{80,320}, b:{1,320} 
layer2 (DenseLayer)       80,80      6 480         W:{80,80}, b:{1,80}                
layer3 (RnnOutputLayer)   80,250     20 250        W:{80,250}, b:{1,250}              
--------------------------------------------------------------------------------------
            Total Parameters:  152 170
        Trainable Parameters:  152 170
           Frozen Parameters:  0
======================================================================================

</pre>

### Training
Training is done by running a java file:
`bg.unisofia.bsmarkov.nlp.TrainNet`
It will parse the corpus and make a data set from each sentence.
Then each sentence is given to the RNN to be fit on it and after that
a `net.rnnClearPreviousState()` is called.

For the training we need the score(error) to be under 1.0 in order this 
RNN to be usable (since its output is one-hot vector).
So 200 epochs are enough, but if you experiment with your own corpus, 
probably you need to adjust epochs and to tweak network architecture.

### Running the prediction algorithm
Predicting the next word we do by running a java file:
`bg.unisofia.bsmarkov.nlp.RunPrediction`
It will parse the file `persist/sentence1.txt` 
and will suggest number of words(similar words) of how eventually 
the sentence can continue.

### Evaluating accuracy
It is quite difficult to estimate accuracy in a real world scenario in any way.
Words can be different, they can even be missing in our lookup table, which is 
quite limited. No one can say if predictions by given single word can be true
or false, since we cannot know that.
Here I suggest to estimate accuracy by measuring similarity(predictedWord,realWord)
from the lookup table built initially with SkipGram.

```text
DataSetSimilarity = SUM(similarity(predictedWord,realWord) for all predictions for all sequences);
AverageDataSetSimilarity = DataSetSimilarity / totalPredictions.
```

You can start java file `EvalNet` and results will be printed similar to that one:
<pre>
EvalNet - Evaluated file: raw_sentences_test.txt
EvalNet - Cumulative similarity: 639,113 for 692 predictions
EvalNet - Average similarity on predictions is 0,924

EvalNet - Evaluated file: raw_sentences_train.txt
EvalNet - Cumulative similarity: 3338,686 for 3429 predictions
EvalNet - Average similarity on predictions is 0,974
 </pre>

You can see that average similarity on trained network on the train set is not even 
perfect, it is `0.974` while the test file gives `0.924`

### Running the prediction engine
You can run java class `bg.unisofia.bsmarkov.nlp.RunPrediction` with your IDE
or with Maven. Then we can see that the samples I gave can be completed quite 
interesting.

Sentence "It would not be the last" becomes "It would not be the last time".
This is in the train data set, so it is learnt. But one can look at the other
examples as well which are not in the learning dataset. I offer also an exhausting
completion till the end of the sentence. 

<pre>
Prediction: "it would not be the last" => [time, season, place]
Exhaustive prediction: "it would not be the last" => [[time, season, place], [., ;]]

Prediction: "the president could" => [use, make, see]
Exhaustive prediction: "the president could" => [[use, make, see], [it, john], [,, --], [times, days], [., ;]]

Prediction: "they are not the" => [people, women, companies]
Exhaustive prediction: "they are not the" => [[people, women, companies], [are, were], [not, also], [of, among], [there, here], [., ;]]

</pre>

Exhaustive completion (or text generation) doesn't make much sense on unseen data.
So just one word prediction makes sense and can be used on apps for chats, etc.
Corpus can be the user input in the chat, and learning algorithm will 
learn the writing habits of the user.

## Possible Improvements
* Algorithm can be improved to take the next probable word from the one-hot output to
make more sensible predictions.

* Another try can be made to represent output vectors as the word2vec encoding used for the input,
of course that would require much training and tuning time from the developer.

* Pre-trained params can be saved from a pre-trained network.
  Later on trainings can fork from that pre-trained network
  and learn new data set easily.


