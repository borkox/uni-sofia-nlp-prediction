package bg.unisofia.bsmarkov.nlp;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.sequencevectors.interfaces.SequenceIterator;
import org.deeplearning4j.models.sequencevectors.iterators.AbstractSequenceIterator;
import org.deeplearning4j.models.sequencevectors.transformers.impl.SentenceTransformer;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.rl4j.policy.ACPolicy;
import org.deeplearning4j.rl4j.space.Box;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.common.util.SerializationUtils;

@Slf4j
public class RunApp {

    public static void main(String[] args) throws IOException {
        BasicLineIterator underlyingIterator = new BasicLineIterator("persist/sentence1.txt");
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());
        SentenceTransformer transformer = new SentenceTransformer.Builder()
                .iterator(underlyingIterator)
                .tokenizerFactory(t)
                .build();
        AbstractSequenceIterator<VocabWord> sequenceIterator =
                new AbstractSequenceIterator.Builder<>(transformer).build();

        ACPolicy<Encodable> policy1 = ACPolicy.load("persist/val1", "persist/pol1");
        ACPolicy<Encodable> policy2 = ACPolicy.load("persist/val1", "persist/pol1");

        policy1.reset();
        policy2.reset();

        WeightLookupTable<VocabWord> lookupTable = SerializationUtils.readObject(new File("persist/lookupTable"));
        SearchVocabCache searchVocabCache1 = new SearchVocabCache(lookupTable, new WordScore1());
        SearchVocabCache searchVocabCache2 = new SearchVocabCache(lookupTable, new WordScore2());

        PlayingNgramMDP mdp1 = new PlayingNgramMDP(searchVocabCache1, sequenceIterator);
        mdp1.reset();
        PlayingNgramMDP mdp2 = new PlayingNgramMDP(searchVocabCache2, sequenceIterator);
        mdp2.reset();

        policy1.play(mdp1);
        policy2.play(mdp2);

        Integer score1 = mdp1.getLastAction();
        Integer score2 = mdp2.getLastAction();

        List<String> suggestions1 = searchVocabCache1.getWordsForScore(score1);
        List<String> suggestions2 = searchVocabCache2.getWordsForScore(score2);
        log.info("Suggestions 1: {}", suggestions1);
        log.info("Suggestions 2: {}", suggestions2);
        log.info("Suggestions intersection: {}", intersect(suggestions1, suggestions2));


    }

    private static Collection<String> intersect(Collection<String> suggestions1, Collection<String> suggestions2) {
        Collection<String> result = new ArrayList<>(suggestions1);
        result.retainAll(suggestions2);
        return result;
    }
}
