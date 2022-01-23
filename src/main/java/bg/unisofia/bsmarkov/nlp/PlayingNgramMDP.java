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

public class PlayingNgramMDP extends LearnNgramMDP {

    public PlayingNgramMDP(SearchVocabCache searchVocabCache,
            AbstractSequenceIterator<VocabWord> sequenceIterator) {
        super(searchVocabCache, sequenceIterator);
    }

    @Override
    public Box reset() {
        this.sequenceIterator.reset();
        return super.reset();
    }

    @Override
    protected int howMuchToSkip() {
        return 0;
    }

    @Override
    protected Sequence<VocabWord> initialSeq() {
        return this.sequenceIterator.nextSequence();
    }
}
