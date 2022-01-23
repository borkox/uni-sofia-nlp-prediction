package bg.unisofia.bsmarkov.nlp;

import java.util.function.Function;

public class WordScore2 implements Function<String , Integer> {

    @Override
    public Integer apply(String word) {
        char firstConsonant= findFirstConsonant(word);
        char secondConsonant = findSecondConsonant(word);

        return Math.min(Math.abs(firstConsonant - secondConsonant), 31);
    }

    private char findFirstConsonant(String word) {
        for (char c : word.toLowerCase().toCharArray()) {
            if (
                    (c !='a') && (c!='e') && (c != 'i')
                            && (c !='o') && (c!='u')) {
                return c;
            }
        }
        return 'b';
    }


    private char findSecondConsonant(String word) {
        int i = 0;
        for (char c : word.toLowerCase().toCharArray()) {
            if (
                    (c !='a') && (c!='e') && (c != 'i')
                            && (c !='o') && (c!='u')) {
                i ++;
                if (i == 2) {
                    return c;
                }
            }
        }
        return 'b';
    }

}
