package bg.unisofia.bsmarkov.nlp;

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
import org.deeplearning4j.models.sequencevectors.transformers.impl.SentenceTransformer;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.rl4j.policy.ACPolicy;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.common.util.SerializationUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMax;
import org.nd4j.linalg.factory.Nd4j;

@Slf4j
public class RunAppWihLstm {

    public static void main(String[] args) throws IOException {
        MultiLayerNetwork net = MultiLayerNetwork.load(new File("persist/rnn"), false);

        WeightLookupTable<VocabWord> lookupTable = SerializationUtils.readObject(new File("persist/lookupTable"));

        SequenceVectors<VocabWord> vectors = new SequenceVectors.Builder<VocabWord>(new VectorsConfiguration())
                .lookupTable(lookupTable).build();

        net.rnnClearPreviousState();

        BasicLineIterator underlyingIterator = new BasicLineIterator("persist/sentence1.txt");
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());
        SentenceTransformer transformer = new SentenceTransformer.Builder()
                .iterator(underlyingIterator)
                .tokenizerFactory(t)
                .build();
        AbstractSequenceIterator<VocabWord> sequenceIterator =
                new AbstractSequenceIterator.Builder<>(transformer).build();
        sequenceIterator.reset();
        List<VocabWord> elements = sequenceIterator.nextSequence().getElements();
        INDArray lastPrediction = null;
        for (VocabWord element : elements) {
            lastPrediction = net
                    .rnnTimeStep(encodeInput(lookupTable, element));
        }
        Collection<String> predicted = decodeFromOutput(vectors, lookupTable, lastPrediction);
        log.info("predicted = " + predicted);


    }

//    private static INDArray encodeInput(WeightLookupTable<VocabWord> lookupTable, VocabWord element) {
//        return lookupTable.vector(element.getWord()).reshape(1, lookupTable.layerSize());
//    }
    private static INDArray encodeInput(WeightLookupTable<VocabWord> lookupTable, VocabWord vocabWord) {
        String word = vocabWord.getWord();
        VocabCache<VocabWord> vocabCache = lookupTable.getVocabCache();
        INDArray indArray = Nd4j.zeros(1, vocabCache.numWords()).putScalar(vocabCache.indexOf(word), 1);
        return indArray;
    }

    private static Collection<String> decodeFromOutput(
            SequenceVectors<VocabWord> vectors,
            WeightLookupTable<VocabWord> lookupTable, INDArray lastPrediction) {

        int wordIndex = Nd4j.getExecutioner().exec(new IMax(lastPrediction, 1)).getInt(0);

//        float aFloat = lastPrediction.argMax(0).getFloat(0);
//        int wordIndex = Float.valueOf(aFloat).intValue();
        log.info("Predicted index: {}", wordIndex);
        String vocabWord = lookupTable.getVocabCache().wordAtIndex(wordIndex);
        return vectors.wordsNearestSum(vocabWord, 5);
    }
}
