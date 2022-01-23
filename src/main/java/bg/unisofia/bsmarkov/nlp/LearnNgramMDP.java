package bg.unisofia.bsmarkov.nlp;

import java.util.List;
import java.util.Random;
import lombok.Getter;
import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.models.sequencevectors.iterators.AbstractSequenceIterator;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.space.Box;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.ObservationSpace;

public class LearnNgramMDP implements MDP<Box, Integer, DiscreteSpace> {

    public static final int SKIP_MAX = 100;
    private final Random random;
    private final SearchVocabCache searchVocabCache;
    private final WordsObservationSpace wordsObservationSpace;
    protected AbstractSequenceIterator<VocabWord> sequenceIterator;
    protected Sequence<VocabWord> vocabWordSequence;
    private int nextWordIndexInSentence;
    private final int[] observationShape;
    @Getter
    private Integer lastAction;

    public LearnNgramMDP(SearchVocabCache searchVocabCache, AbstractSequenceIterator<VocabWord> sequenceIterator) {
        this.wordsObservationSpace = new WordsObservationSpace(searchVocabCache);
        this.sequenceIterator = sequenceIterator;
        this.random = new Random(System.currentTimeMillis());
        this.searchVocabCache = searchVocabCache;
        observationShape = new int[]{1, this.searchVocabCache.vectorLen(), 1};
    }

    @Override
    public ObservationSpace<Box> getObservationSpace() {
        return wordsObservationSpace;
    }

    @Override
    public DiscreteSpace getActionSpace() {
        return new DiscreteSpace(32);
    }

    @Override
    public Box reset() {
        this.nextWordIndexInSentence = 1;
        int skip = howMuchToSkip();
        vocabWordSequence = initialSeq();
        for (int i = 0; i < skip; i++) {
            if (!this.sequenceIterator.hasMoreSequences()) {
                // if we reached end of the sentences we start again
                this.sequenceIterator.reset();
            }
            vocabWordSequence = this.sequenceIterator.nextSequence();
        }
        if (vocabWordSequence == null) {
            throw new RuntimeException("Cannot make next observation");
        }
        String firstWord = vocabWordSequence.getElements().get(0).getWord();
        WordObservation observation = new WordObservation(searchVocabCache.vector(firstWord), firstWord);
        return new Box(observationShape, observation.toArray());
    }

    protected Sequence<VocabWord> initialSeq() {
        return null;
    }

    protected int howMuchToSkip() {
        return 1 + random.nextInt(SKIP_MAX);
    }

    @Override
    public void close() {
        this.sequenceIterator.reset();
    }

    @Override
    public StepReply<Box> step(Integer integer) {
        this.lastAction = integer;
        List<String> predictedWordsForScore = this.searchVocabCache.getWordsForScore(integer);

//        String predicted = lookupTable.getVocabCache().wordAtIndex(integer - 1);
        VocabWord elementByIndex = vocabWordSequence.getElementByIndex(nextWordIndexInSentence ++);
        double reward = 0;
        String realWord = elementByIndex.getWord();
        if (predictedWordsForScore.contains(realWord)) {
            reward = 0.1;
        }
        WordObservation observation = new WordObservation(searchVocabCache.vector(realWord), realWord);
        return new StepReply<>(new Box(this.observationShape, observation.toArray()), reward, isDone(), null);
    }

    @Override
    public boolean isDone() {
        return nextWordIndexInSentence >= this.vocabWordSequence.getElements().size();
    }

    @Override
    public MDP<Box, Integer, DiscreteSpace> newInstance() {
        return new LearnNgramMDP(searchVocabCache, sequenceIterator);
    }
}
