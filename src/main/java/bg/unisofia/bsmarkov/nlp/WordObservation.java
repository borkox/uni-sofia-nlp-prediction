package bg.unisofia.bsmarkov.nlp;

import lombok.AllArgsConstructor;
import org.deeplearning4j.rl4j.space.Encodable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class WordObservation implements Encodable {

    private final INDArray array;
    private final String word;

    public WordObservation(INDArray array, String word) {
        this.array = array;
        this.word = word;
    }

    @Override
    public double[] toArray() {
        return array.toDoubleVector();
    }

    @Override
    public boolean isSkipped() {
        return false;
    }

    @Override
    public INDArray getData() {
        return array;
    }

    @Override
    public Encodable dup() {
        return new WordObservation(array, word);
    }
}
