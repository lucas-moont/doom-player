# Study guides

One guide per milestone, named after it (`M0-workshop.md`). A guide turns agent-written code into something a beginner can read, understand, and rebuild.

The reader knows basic Python (variables, functions, loops) and is new to machine learning.

## Every guide has these sections, in this order

1. **What was built**: three sentences, plain words.
2. **Concepts**: each new concept with an everyday analogy first, then the real term, then one link to a primary source.
3. **Reading order**: a numbered list of files (and line ranges for long files), starting where execution starts. Each entry says what to look for and what to skip on first read.
4. **Follow one Attempt**: a walk through the code path of a single Attempt, naming each function it passes through.
5. **Try it**: three to five small experiments that change one value and predict the outcome before running ("halve the frame skip: what happens to Attempt length?").
6. **Rebuild it**: the exercise for the study repository. States the smaller version to build by hand, what is given, and how to check it works. Contains no solution code.
7. **Check yourself**: five questions answerable only by someone who understood the milestone.

## Completion criteria for a guide

- Every file in the milestone's diff appears in the reading order, or is listed under "safe to skip" with a reason
- Every term in the guide that is in `CONTEXT.md` is used with its glossary meaning
- The "Rebuild it" exercise fits in three study sessions of 2-3 hours
