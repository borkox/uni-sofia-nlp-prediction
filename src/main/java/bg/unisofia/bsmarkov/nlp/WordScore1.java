package bg.unisofia.bsmarkov.nlp;

import java.util.function.Function;

public class WordScore1 implements Function<String , Integer> {

    @Override
    public Integer apply(String word) {
        char firstVowel = findFirstVowel(word);
        char firstConsonant = findFirstConsonant(word);

        return Math.min(Math.abs(firstVowel - firstConsonant), 31);
    }

    private char findFirstVowel(String word) {
        for (char c : word.toLowerCase().toCharArray()) {
            if (
                    (c =='a') || (c=='e') || (c == 'i')
                            || (c =='o') || (c=='u')) {
                return c;
            }
        }
        return 'a';
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

}
