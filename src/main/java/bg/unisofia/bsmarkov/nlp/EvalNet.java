package bg.unisofia.bsmarkov.nlp;

import java.io.File;
import java.io.IOException;
import java.util.List;
import javax.annotation.Nullable;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.sequencevectors.SequenceVectors;
import org.deeplearning4j.models.sequencevectors.iterators.AbstractSequenceIterator;
import org.deeplearning4j.models.sequencevectors.transformers.impl.SentenceTransformer;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.LowCasePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.common.util.SerializationUtils;
import org.nd4j.linalg.api.ndarray.INDArray;

@Slf4j
public class EvalNet {

    public static final String TEST_CORPUS = "/raw_sentences_test.txt";
    public static final String TRAIN_CORPUS = "/raw_sentences_train.txt";

    public static void main(String[] args) throws IOException {
        evalCorpus(TEST_CORPUS);
        evalCorpus(TRAIN_CORPUS);

    }

    private static void evalCorpus(String testCorpus) throws IOException {
        String pathname = TrainNet.class.getResource(testCorpus).getFile();
        log.info("Reading from pathname: {}", pathname);
        File file = new File(pathname);

        MultiLayerNetwork net = MultiLayerNetwork.load(new File("persist/rnn"), false);

        WeightLookupTable<VocabWord> lookupTable = SerializationUtils.readObject(new File("persist/lookupTable"));

        SequenceVectors<VocabWord> vectors = new SequenceVectors.Builder<VocabWord>(new VectorsConfiguration())
                .lookupTable(lookupTable).build();

        BasicLineIterator underlyingIterator = new BasicLineIterator(file);
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new LowCasePreProcessor());
        SentenceTransformer transformer = new SentenceTransformer.Builder()
                .iterator(underlyingIterator)
                .tokenizerFactory(t)
                .build();
        AbstractSequenceIterator<VocabWord> sequenceIterator =
                new AbstractSequenceIterator.Builder<>(transformer).build();
        sequenceIterator.reset();

        double cumulativeSimilarity = 0;
        int totalPredictions = 0;
        while (sequenceIterator.hasMoreSequences()) {
            // start fresh sequence by clearing previous state
            net.rnnClearPreviousState();

            List<VocabWord> elements = sequenceIterator.nextSequence().getElements();
            INDArray lastPrediction;
            for (int i = 0; i < elements.size() - 1; i++) {
                VocabWord currentWord = elements.get(i);
                VocabWord nextWord = elements.get(i + 1);
                lastPrediction = net.rnnTimeStep(EncodeDecodeUtils.encodeInput(lookupTable, currentWord.getWord()));
                @Nullable
                String predictedWord = EncodeDecodeUtils.decodeOutputOneHotToWord(vectors, lookupTable, lastPrediction);
                double similarity = vectors.similarity(nextWord.getWord(), predictedWord);
                cumulativeSimilarity += similarity;
                totalPredictions ++;
            }
        }

        log.info("Evaluated file: {}", file.getName());
        log.info("Cumulative similarity: {} for {} predictions", String.format("%.3f", cumulativeSimilarity), totalPredictions);
        log.info("Average similarity on predictions is {}", String.format("%.3f", cumulativeSimilarity/ totalPredictions));
    }

}
