# Imparting Interpretability to Word Embeddings While Preserving Semantic Structure 

In this study, we introduce an additive modification to the objective function of the GloVe embedding learning algorithm that encourages the embedding vectors of words that are semantically related to a predefined concept to take larger values along a specified dimension, while leaving the original semantic learning mechanism mostly unaffected. In other words, we align words that are already determined to be related, along predefined concepts. Therefore, we impart interpretability to the word embedding by assigning meaning to its vector dimensions. The predefined concepts are derived from an external lexical resource, Roget’s Thesaurus. It is observed that alignment along the chosen concepts is not limited to words in the Thesaurus and extends to other related words as well. Extent of the interpretability and preservation of semantic coherence of resulting vectors can be observed via various evaluation methods.

## Training

In order to train, run ``` imparting_interpretability_demo.sh ``` with corpus, available RAM memory, vector size (number of embedding dimensions ) and number of threads information. For example;


```bash
bash imparting_interpretability_demo.sh /path/to/your/corpus 240.0 300 45
```
This demo will initialize training with k = 0.1 and every dimension will be imparted along positive direction. You can change hyperparameters from demo file and from the files in Params folder. Additionally, you can download [pretrained word vectors with k = 0.1 parameter and 300 dimension](https://drive.google.com/file/d/1hpWT3Vc_-JTuPDeYgL5FZAPcSF2fEt6f/view?usp=sharing) trained on a snapshot of English Wikipedia consisting of around 1.1B tokens, with the stop-words filtered out. 

## Evaluation 

You can make analogy, interpretabilty, sentiment analysis and word similarity tests by running respective commands on terminal.

```
python analogy_eval.py --vectors_file /path/to/vector/file --output_file analogy_result.txt
python interpretability_eval.py --vectors_file /path/to/vector/file
python sentiment_eval.py --vectors_file /path/to/vector/file --output_file sentiment_result.txt
bash evaluate_word_similarity.sh /path/to/vector/file
```
### Word Intrusion Test

Word intrusion test is a multiple choice test where each choice is a separate word. Four of these words are chosen among the words whose vector values at a specific dimension are high and one is chosen from the words whose vector values are low at that specific dimension. This word is called an intruder word. If the participant can distinguish the intruder word from others, it can be said that this dimension is interpretable. We both used imparted and original GloVe embeddings for comparison. For each dimension of both embeddings, we prepare a question. We shuffled the questions in random order so that participant cannot know which question comes from which embedding. In total, there are 600 questions (300 GloVe + 300 imparted GloVe) with five choices for each. Test can be started by running ``` word_intrusion_offline.py ```. At the end of the test, answers and scores of the participant will be saved as pickle files.

### Requirements for Evaluation
* numpy
* sklearn

## Citation

If you make use of these tools, please cite following paper.

```
@article{şenel_utlu_şahinuç_ozaktas_koç_2020, 
  title={Imparting interpretability to word embeddings while preserving semantic structure}, 
  author={Şenel, Lütfi Kerem and Utlu, İhsan and Şahinuç, Furkan and Ozaktas, Haldun M. and Koç, Aykut},
  DOI={10.1017/S1351324920000315}, 
  journal={Natural Language Engineering}, 
  publisher={Cambridge University Press}, 
  year={2020}, 
  pages={1–26}}
```

## Contact Info
For every issues related to code and implementation please contact aykutkoc.lab@gmail.com
