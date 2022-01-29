package bg.unisofia.bsmarkov.nlp;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.sequencevectors.SequenceVectors;
import org.deeplearning4j.models.sequencevectors.iterators.AbstractSequenceIterator;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.transformers.impl.SentenceTransformer;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.VocabConstructor;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.LowCasePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.common.util.SerializationUtils;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

@Slf4j
public class TrainNet {

    public static final int TRAIN_EPOCHS = 200;
    public static final int PRINT_ITERATIONS = 10;
    public static final int PRINT_PROCESSING_SENTENCES = 30;
    public static final String TRAIN_CORPUS = "/raw_sentences_train.txt";
    public static final double LEARNING_RATE = 0.005;
    public static final double L2_REGULARIZATION = 0.000001;
    public static final int LAYER_SIZE = 80;

    public static void main(String[] args) throws IOException {
        String pathname = TrainNet.class.getResource(TRAIN_CORPUS).getFile();
        log.info("Reading from pathname: {}", pathname);
        File file = new File(pathname);

        /*
            First we build line iterator
         */
        BasicLineIterator underlyingIterator = new BasicLineIterator(file);


        /*
            Now we need the way to convert lines into Sequences of VocabWords.
            In this example that's SentenceTransformer
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

        WeightLookupTable<VocabWord> lookupTable = SerializationUtils.readObject(new File("persist/lookupTable"));

        VocabCache<VocabWord> vocabCache = lookupTable.getVocabCache();
        /*
            Now we should build vocabulary out of sequence iterator.
            We can skip this phase, and just set AbstractVectors.resetModel(TRUE), and vocabulary will be mastered internally
        */
        VocabConstructor<VocabWord> constructor = new VocabConstructor.Builder<VocabWord>()
                .addSource(sequenceIterator, 0)
                .setTargetVocabCache(vocabCache)
                .build();

        constructor.buildJointVocabulary(false, true);


        SequenceVectors<VocabWord> vectors = new SequenceVectors.Builder<VocabWord>(new VectorsConfiguration())
                .lookupTable(lookupTable).build();

        log.info("Test similarity: (day,night) = {}", vectors.similarity("day", "night"));
        log.info("Test words nearest: (day) = {}", vectors.wordsNearest("day", 3));

        //Set up network configuration:
        int inputVectorSz = lookupTable.layerSize();
        int outputVectorSz = vocabCache.numWords();

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .l2(L2_REGULARIZATION)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(LEARNING_RATE))
                .list()
                .layer(new LSTM.Builder().nIn(inputVectorSz).nOut(LAYER_SIZE)
                        .activation(Activation.TANH).build())
                .layer(new LSTM.Builder().nIn(LAYER_SIZE).nOut(LAYER_SIZE)
                        .activation(Activation.TANH).build())
                .layer(new DenseLayer.Builder().nIn(LAYER_SIZE).nOut(LAYER_SIZE)
                        .activation(Activation.RELU).build())
                .layer(new RnnOutputLayer.Builder(LossFunction.MCXENT).activation(Activation.SOFTMAX)
                        .nIn(LAYER_SIZE).nOut(outputVectorSz).build())
                .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(50)
                .tBPTTBackwardLength(50)
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(PRINT_ITERATIONS));

        //Print the  number of parameters in the network (and for each layer)
        log.info("Summary: {} ", net.summary());

        List<DataSet> datasetList = new ArrayList<>();
        sequenceIterator.reset();
        int sequence = 0;
        while (sequenceIterator.hasMoreSequences()) {
            Sequence<VocabWord> vocabWordSequence = sequenceIterator.nextSequence();
            List<VocabWord> elements = vocabWordSequence.getElements();

            // If we have sentence of N words, then we can predict N-1 words knowing the first word.
            // That is why we have features vector of size "elements.size()-1"
            INDArray features = Nd4j.zeros(elements.size()-1, inputVectorSz);
            INDArray labels  = Nd4j.zeros(elements.size()-1, outputVectorSz);
            for (int i = 0; i < elements.size() - 1; i++) {
                VocabWord vocabWord = elements.get(i);
                VocabWord nextWord = (i + 1 < elements.size()) ? elements.get(i + 1) : null;

                features.putRow(i, encodeInput(lookupTable, vocabWord));
                labels.putRow(i, encodeOutput(lookupTable, nextWord));
            }
            INDArray reshapedFeatures = features.transpose().reshape(1,  inputVectorSz, elements.size() - 1);
            INDArray reshapedLabels = labels.transpose().reshape(1, outputVectorSz, elements.size() - 1);
            DataSet ds = new DataSet(
                    reshapedFeatures,
                    reshapedLabels);

            datasetList.add(ds);
            sequence ++;
            if (sequence % PRINT_PROCESSING_SENTENCES == 0) {
                log.info("Processing sentence {}", sequence);
            }
        }
        log.info("Start learn network");
        for(int i=0 ; i < TRAIN_EPOCHS; i ++) {
            log.info("Epoch: {}", i);
            Collections.shuffle(datasetList);
            for (DataSet dataSet : datasetList) {
                net.fit(dataSet);
                // clear current stance from the last example
                net.rnnClearPreviousState();
            }
        }
        net.save(new File("persist/rnn"));

    }

    /**
     * Output vectors are on-hot vectors.
     */
    private static INDArray encodeOutput(WeightLookupTable<VocabWord> lookupTable, VocabWord vocabWord) {
        String word = vocabWord.getWord();
        VocabCache<VocabWord> vocabCache = lookupTable.getVocabCache();
        return Nd4j.zeros(1, vocabCache.numWords()).putScalar(vocabCache.indexOf(word), 1);
    }

    /**
     * Input vectors are according to word2vec algorithm.
     */
    private static INDArray encodeInput(WeightLookupTable<VocabWord> lookupTable, VocabWord vocabWord) {
        return lookupTable.vector(vocabWord.getWord());
    }
}
