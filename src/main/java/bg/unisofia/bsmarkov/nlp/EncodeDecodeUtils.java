package bg.unisofia.bsmarkov.nlp;

import java.util.Collection;
import javax.annotation.Nullable;
import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.sequencevectors.SequenceVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMax;
import org.nd4j.linalg.factory.Nd4j;

@NoArgsConstructor(access = AccessLevel.PRIVATE)
@Slf4j
public final class EncodeDecodeUtils {

    @Nullable
    static String decodeOutputOneHotToWord(
            SequenceVectors<VocabWord> vectors,
            WeightLookupTable<VocabWord> lookupTable, INDArray lastPrediction) {

        int wordIndex = Nd4j.getExecutioner().exec(new IMax(lastPrediction, 1)).getInt(0);
        String vocabWord = lookupTable.getVocabCache().wordAtIndex(wordIndex);
        Collection<String> words = vectors.wordsNearestSum(vocabWord, 1);
        if (words.isEmpty()) {
            return null;
        }
        return words.iterator().next();
    }

    static Collection<String> decodeOutputOneHotToNearestWords(
            SequenceVectors<VocabWord> vectors,
            WeightLookupTable<VocabWord> lookupTable, INDArray lastPrediction, int howMany) {

        int wordIndex = Nd4j.getExecutioner().exec(new IMax(lastPrediction, 1)).getInt(0);
        String vocabWord = lookupTable.getVocabCache().wordAtIndex(wordIndex);
        return vectors.wordsNearestSum(vocabWord, howMany);
    }

    static INDArray encodeInput(WeightLookupTable<VocabWord> lookupTable, String word) {
        return lookupTable.vector(word).reshape(1, lookupTable.layerSize());
    }
}
