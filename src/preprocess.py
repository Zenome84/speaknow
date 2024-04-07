
import pandas as pd
import numpy as np

def get_audio_metadata(location):
    speaknow = pd.read_csv(location)

    metadata = pd.DataFrame()
    metadata['assessment_id'] = np.concatenate([
        speaknow['assessment_id'],
        speaknow['assessment_id'],
        speaknow['assessment_id'],
        speaknow['assessment_id'],
        speaknow['assessment_id']
    ], 0)
    metadata['file_name'] = np.concatenate([
        './data/' + speaknow['assessment_id'].astype(str) + '-1.mp3',
        './data/' + speaknow['assessment_id'].astype(str) + '-2.mp3',
        './data/' + speaknow['assessment_id'].astype(str) + '-3.mp3',
        './data/' + speaknow['assessment_id'].astype(str) + '-4.mp3',
        './data/' + speaknow['assessment_id'].astype(str) + '-5.mp3'
    ], 0)

    return metadata

if __name__ == "__main__":
    from pronunciation import PronunciationEvaluator

    metadata = get_audio_metadata("./SpeakNow_test_data.csv")

    scores, confidences = PronunciationEvaluator()(
        metadata.file_name.values.tolist(),
        metadata.assessment_id.values.tolist(),
        score_type='mean',
        confidence_type='std'
    )

    print(scores)