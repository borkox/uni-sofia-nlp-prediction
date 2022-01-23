package bg.unisofia.bsmarkov.nlp;

import java.io.File;
import java.io.IOException;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.sequencevectors.SequenceVectors;
import org.deeplearning4j.models.sequencevectors.iterators.AbstractSequenceIterator;
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
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.common.util.SerializationUtils;
import org.nd4j.linalg.learning.config.Adam;

@Slf4j
public class TrainWithScoring2 {

    public static A3CDiscrete.A3CConfiguration A3C =
            new A3CDiscrete.A3CConfiguration(
                    123,            //Random seed
                    10000,          //Max step By epoch
                    100_000,        //Max step
                    16,              //Number of threads
                    16,             //t_max
                    500,            //num step noop warmup
                    0.5,            //reward scaling
                    0.8,           //gamma
                    10.0            //td-error clipping
            );
    private static final ActorCriticFactorySeparateStdDense.Configuration NET_A3C = ActorCriticFactorySeparateStdDense.Configuration
            .builder()
            .useLSTM(true)
            .updater(new Adam(1e-3))
            .l2(0)
            .numHiddenNodes(40)
            .numLayer(2)
            .build();

    public static void main(String[] args) throws IOException {
        String pathname = TrainWithScoring2.class.getResource("/raw_sentences.txt").getFile();
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
         */
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

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


        WeightLookupTable<VocabWord> lookupTable = SerializationUtils.readObject(new File("persist/lookupTable"));

        SequenceVectors<VocabWord> vectors = new SequenceVectors.Builder<VocabWord>(new VectorsConfiguration())
                .lookupTable(lookupTable).build();

        log.info("Test similarity: (day,night) = {}", vectors.similarity("day", "night"));
        log.info("Test words nearest: (day) = {}", vectors.wordsNearest("day", 3));

        LearnNgramMDP learnNgramMDP = new LearnNgramMDP(new SearchVocabCache(lookupTable, new WordScore2()),
                sequenceIterator);
        DataManager manager = new DataManager(false);

        //define the training
        A3CDiscreteDense<Box> a3c = new A3CDiscreteDense<>(learnNgramMDP, NET_A3C, A3C, manager);

        //start the training with reinforcement learning
        a3c.train();

        ACPolicy<Box> pol = a3c.getPolicy();

        pol.save("persist/val2", "persist/pol2");

        //close the mdp (http connection)
        learnNgramMDP.close();

    }
}
