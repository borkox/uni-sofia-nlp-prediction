package bg.unisofia.bsmarkov.nlp;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.learning.impl.elements.SkipGram;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.sequencevectors.SequenceVectors;
import org.deeplearning4j.models.sequencevectors.iterators.AbstractSequenceIterator;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.transformers.impl.SentenceTransformer;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabConstructor;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.rl4j.learning.async.a3c.discrete.A3CDiscrete;
import org.deeplearning4j.rl4j.learning.async.a3c.discrete.A3CDiscreteDense;
import org.deeplearning4j.rl4j.network.ac.ActorCriticFactorySeparateStdDense;
import org.deeplearning4j.rl4j.policy.ACPolicy;
import org.deeplearning4j.rl4j.space.Box;
import org.deeplearning4j.rl4j.util.DataManager;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.LowCasePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.common.util.SerializationUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.learning.config.Adam;

@Slf4j
public class MakeLookupTable {

    public static final int VECTOR_LEN = 150;

    public static void main(String[] args) throws IOException {
        String pathname = MakeLookupTable.class.getResource("/raw_sentences.txt").getFile();
        log.info("Reading from pathname: {}", pathname);
        File file = new File(pathname);

        AbstractCache<VocabWord> vocabCache = new AbstractCache.Builder<VocabWord>().build();

        /*
            First we build line iterator
         */
        BasicLineIterator underlyingIterator = new BasicLineIterator(file);


        /*
            Now we need the way to convert lines into Sequences of VocabWords.
            In this example that's SentenceTransformer
            For preprocessing we only put lowercase.
         */
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new LowCasePreProcessor());

        SentenceTransformer transformer = new SentenceTransformer.Builder()
                .iterator(underlyingIterator)
                .tokenizerFactory(t)
                .build();


        /*
            And we pack that transformer into AbstractSequenceIterator
         */
        AbstractSequenceIterator<VocabWord> sequenceIterator =
                new AbstractSequenceIterator.Builder<>(transformer).build();


        /*
            Now we should build vocabulary out of sequence iterator.
            We can skip this phase, and just set AbstractVectors.resetModel(TRUE), and vocabulary will be mastered internally
        */
        VocabConstructor<VocabWord> constructor = new VocabConstructor.Builder<VocabWord>()
                .addSource(sequenceIterator, 0)
                .setTargetVocabCache(vocabCache)
                .build();

        constructor.buildJointVocabulary(false, true);

        /*
            Time to build WeightLookupTable instance for our new model
        */

        WeightLookupTable<VocabWord> lookupTable = new InMemoryLookupTable.Builder<VocabWord>()
                .vectorLength(VECTOR_LEN)
                .useAdaGrad(false)
                .cache(vocabCache)
                .build();

        sequenceIterator.reset();
        lookupTable.resetWeights(true);

        /*
            Now we can build AbstractVectors model, that suits our needs
         */
        SequenceVectors<VocabWord> vectors = new SequenceVectors.Builder<VocabWord>(new VectorsConfiguration())
                // minimum number of occurrences for each element in training corpus. All elements below this value will be ignored
                // Please note: this value has effect only if resetModel() set to TRUE, for internal model building. Otherwise it'll be ignored, and actual vocabulary content will be used
                .minWordFrequency(1)

                // WeightLookupTable
                .lookupTable(lookupTable)

                // abstract iterator that covers training corpus
                .iterate(sequenceIterator)

                // vocabulary built prior to modelling
                .vocabCache(vocabCache)

                // batchSize is the number of sequences being processed by 1 thread at once
                // this value actually matters if you have iterations > 1
                .batchSize(250)

                // number of iterations over batch
                .iterations(1)

                // number of iterations over whole training corpus
                .epochs(1)

                // if set to true, vocabulary will be built from scratches internally
                // otherwise externally provided vocab will be used
                .resetModel(false)

                /*
                    These two methods define our training goals. At least one goal should be set to TRUE.
                 */
                .trainElementsRepresentation(true)
                .trainSequencesRepresentation(false)

                /*
                    Specifies elements learning algorithms. SkipGram, for example.
                 */
                .elementsLearningAlgorithm(new SkipGram<VocabWord>())
                .windowSize(2)
                .build();

        /*
            Now, after all options are set, we just call fit()
         */
        vectors.fit();

        log.info("Total words : {}", lookupTable.getVocabCache().numWords());

        log.info("Scaling vectors between 0,1");
        INDArray weights = lookupTable.getWeights();
        DataSet datasetForScale = new DataSet();
        datasetForScale.setFeatures(weights);
        NormalizerMinMaxScaler scaler = new NormalizerMinMaxScaler(0,1);
        scaler.fit(datasetForScale);
        log.info("MAX vector: {}", scaler.getMax());
        log.info("MIN vector: {}", scaler.getMin());
        scaler.transform(weights);

        log.info("Test similarity: (day,night) = {}", vectors.similarity("day", "night"));
        log.info("Test words nearest: (day) = {}", vectors.wordsNearest("day", 3));

        SerializationUtils.saveObject(lookupTable, new File("persist/lookupTable"));
    }
}
