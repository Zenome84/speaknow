## SpeakNow Home Assignment - ToC

This body of work is designed to demo different coding style, knowledge, and analysis. It contains:
* A PowerPoint report summarizing the Home Assignment is in [report.pptx](./report.pptx)
    * In the report I also discuss scaling these results.
    * Can also be opened with Google Slides.
* Under [src](./src) you will find the code to run inference on the scoring models:
    * [evaluator.py](./src/evaluator.py): Parent class to run Whisper fine-tuned models - containing generic rating and confidence measures.
    * [fluency.py](./src/fluency.py): Fluency rating.
    * [pronunciation.py](./src/pronunciation.py): Pronunciation rating.
    * [vocabulary.py](./src/vocabulary.py): Vocabulary rating.
        * To run this you must download and unzip the trained model [whisper-medium-vocabulary-accuracy.7z](https://1drv.ms/u/s!AjZC9GwJFmuHgs9ryeSYPIle-WBmEA?e=mSBN1X) into the main directory.
    * [preprocess.py](./src/preprocess.py): helper function.
* In [data_analysis.ipynb](./data_analysis.ipynb) you will find details related to the DA section of the report.
* In [model_analysis.ipynb](./model_analysis.ipynb) you will find the details related to results of running fluency/pronunciation/vocabulary rating models.
* In [finetune_whisper_vocabulary](./finetune_whisper_vocabulary.py) you will find the script to fine-tune Whisper on the vocabulary data.
    * I have excluded all SpeakNow data from this project. To run any of these, you will need to add the mp3 files to the [data](./data/) folder and `SpeakNow_test_data.csv` to the main directory.
* In [cohesion.ipynb] you will find two things:
    * Transcription of all the audio segments using Whisper.
    * A test for a cohesion rating model with interpretability.
    * Analysis of results.
* Finally, I add that I did not put together anything for grammar, yet.
