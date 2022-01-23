package bg.unisofia.bsmarkov.nlp;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.iterator.Word2VecDataSetIterator;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.Nd4jCpu.NDArray;

@Slf4j
public class SearchVocabCache {

    private final int numWords;
    private final WeightLookupTable<VocabWord> lookupTable;
    private Map<String, Integer> map = new HashMap<>();
    private Map<Integer, List<String>> reverseMap = new HashMap<>();

    public SearchVocabCache(WeightLookupTable<VocabWord> lookupTable, Function<String, Integer> wordEmbedding) {

        this.lookupTable = lookupTable;
        VocabCache<VocabWord> vocabCache = lookupTable.getVocabCache();
        numWords = vocabCache.numWords();

        for (Iterator<VocabWord> iterator = vocabCache.vocabWords().iterator(); iterator.hasNext(); ) {
            VocabWord vocabWord = iterator.next();
            String word = vocabWord.getWord();

            int score = wordEmbedding.apply(word);

            System.out.println("score for " + word + " = " + score);
            map.put(word, score);
            reverseMap.computeIfAbsent(score, (a) -> new ArrayList<>());
            List<String> wordsForThatIndex = reverseMap.get(score);
            if (!wordsForThatIndex.contains(word)) {
                wordsForThatIndex.add(word);
            }

        }

//        INDArray weights = lookupTable.getWeights();
//
//        DataSet datasetForScale = new DataSet();
//        datasetForScale.setFeatures(weights);
//        NormalizerMinMaxScaler scaler = new NormalizerMinMaxScaler();
//        scaler.fit(datasetForScale);
//
//        log.info("MAX vector: {}", scaler.getMax());
//        log.info("MIN vector: {}", scaler.getMin());
//
//        scaler.transform(weights);
//
//        log.info("Ready with scaling");
    }

    public List<String> getWordsForScore(int score) {
        return reverseMap.getOrDefault(score, List.of());
    }


    public int getScoreForWord(String realWord) {
        return map.getOrDefault(realWord, 0);
    }

    public INDArray vector(String realWord) {
        return lookupTable.vector(realWord);
    }
    public int vectorLen() {
        return lookupTable.layerSize();
    }
}
