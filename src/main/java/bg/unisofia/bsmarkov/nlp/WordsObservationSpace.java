package bg.unisofia.bsmarkov.nlp;

import lombok.AllArgsConstructor;
import org.deeplearning4j.rl4j.space.Box;
import org.deeplearning4j.rl4j.space.ObservationSpace;
import org.nd4j.linalg.api.ndarray.INDArray;

@AllArgsConstructor
public class WordsObservationSpace implements ObservationSpace<Box> {

    private final int wordRepresentationVectorLen;

    public WordsObservationSpace(SearchVocabCache searchVocabCache) {
        this.wordRepresentationVectorLen = searchVocabCache.vectorLen();
    }

    @Override
    public String getName() {
        return "Box(1,1," + wordRepresentationVectorLen+")";
    }

    @Override
    public int[] getShape() {
        return new int[] {1, 1, wordRepresentationVectorLen};
    }

    @Override
    public INDArray getLow() {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray getHigh() {
        throw new UnsupportedOperationException();
    }

}
