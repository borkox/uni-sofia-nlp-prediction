package bg.unisofia.bsmarkov.nlp;

import static bg.unisofia.bsmarkov.nlp.EncodeDecodeUtils.encodeInput;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.sequencevectors.SequenceVectors;
import org.deeplearning4j.models.sequencevectors.iterators.AbstractSequenceIterator;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
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
public class RunPrediction {

    public static final String SAVED_NETWORK = "persist/rnn";
    public static final String SAVED_LOOKUP_TABLE = "persist/lookupTable";
    public static final String EXAMPLE_SEQUENCE_FILE = "persist/sentence1.txt";

    public static void main(String[] args) throws IOException {
        MultiLayerNetwork net = MultiLayerNetwork.load(new File(SAVED_NETWORK), false);

        WeightLookupTable<VocabWord> lookupTable = SerializationUtils.readObject(new File(SAVED_LOOKUP_TABLE));

        SequenceVectors<VocabWord> vectors = new SequenceVectors.Builder<VocabWord>(new VectorsConfiguration())
                .lookupTable(lookupTable).build();

        net.rnnClearPreviousState();

        BasicLineIterator underlyingIterator = new BasicLineIterator(EXAMPLE_SEQUENCE_FILE);
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new LowCasePreProcessor());
        SentenceTransformer transformer = new SentenceTransformer.Builder()
                .iterator(underlyingIterator)
                .tokenizerFactory(t)
                .build();
        AbstractSequenceIterator<VocabWord> sequenceIterator =
                new AbstractSequenceIterator.Builder<>(transformer).build();
        sequenceIterator.reset();
        while (sequenceIterator.hasMoreSequences()) {
            Sequence<VocabWord> sequence = sequenceIterator.nextSequence();
            List<VocabWord> elements = sequence.getElements();
            INDArray lastPrediction = null;
            for (VocabWord element : elements) {
                lastPrediction = net
                        .rnnTimeStep(encodeInput(lookupTable, element.getWord()));
            }
            Collection<Collection<String>> exhaustingPredictions = new ArrayList<>();
            Collection<String> predicted = EncodeDecodeUtils
                    .decodeOutputOneHotToNearestWords(vectors, lookupTable, lastPrediction, 3);
            exhaustingPredictions.add(predicted);
            exhaustivePrediction(net, lookupTable, vectors, exhaustingPredictions, predicted);
            log.info("Prediction: \"{}\" => {}", String.join(" ", sequence.asLabels()), predicted);
            log.info("Exhaustive prediction: \"{}\" => {}", String.join(" ", sequence.asLabels()), exhaustingPredictions);
        }

    }

    private static void exhaustivePrediction(MultiLayerNetwork net, WeightLookupTable<VocabWord> lookupTable,
            SequenceVectors<VocabWord> vectors, Collection<Collection<String>> exhaustingPredictions,
            Collection<String> predicted) {
        INDArray lastPrediction;
        while (!predicted.isEmpty()
                && !(predicted.contains(".") || predicted.contains("?") || predicted.contains("!"))
        ) {
            lastPrediction = net.rnnTimeStep(encodeInput(lookupTable, predicted.iterator().next()));
            predicted = EncodeDecodeUtils
                    .decodeOutputOneHotToNearestWords(vectors, lookupTable, lastPrediction, 2);
            exhaustingPredictions.add(predicted);
        }
    }

}
